"""
Train and honestly evaluate a bowl-depth regressor on metric MoGe features.

What is different from train_real_labels.py
-------------------------------------------
1. Features keep their millimetres (metric_features.py) instead of passing
   through features.py's per-image min-max normalisation.
2. Repeated grouped K-fold over CAPTURE SESSIONS (frames within 60 s of each other,
   often the same stretch of road or the same pothole twice) instead of one split,
   so every number comes with a spread. One 25% split of ~250 rows moves by
   around half a millimetre from seed to seed, which is the size of the
   differences we are trying to judge.
3. Every fold is scored three times on the SAME held-out frames: with the human
   polygon, with the production YOLO mask, and with the SAM 2-refined mask. The
   human-mask score is what the model can do; the YOLO/SAM 2 score is what the
   served pipeline would do.
4. Baselines (median, log area) are refitted inside every fold.

Outputs
-------
    ml_results/pothrgbd/metric_regressor_results.json   every variant, mean and spread
    ml_models/metric/depth_regressor.joblib             the chosen model, fitted on all frames
    ml_models/metric/depth_regressor_meta.json          features, CV scores, error-bar quantiles

Usage:
    python scripts/train_metric_regressor.py
"""
import json
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from metric_features import FEATURE_COLS, SHAPE_COLS           # noqa: E402
from scripts.pothrgbd_metric_labels import capture_sessions, severity_from_mm  # noqa: E402

DATA = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
OLD = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_real_labels.csv")
RESULTS = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "metric_regressor_results.json")
MODEL_DIR = os.path.join(PROJECT_DIR, "ml_models", "metric")

K, REPEATS, SEED = 5, 3, 20260924
SOURCES = ("gt", "yolo", "sam2")


def folds(keys):
    """Repeated grouped K-fold over group ids (capture sessions)."""
    uniq = np.array(sorted(set(keys)))
    for r in range(REPEATS):
        rng = np.random.default_rng(SEED + r)
        perm = rng.permutation(uniq)
        for f in range(K):
            test = set(perm[f::K])
            yield r, f, test


def models():
    return {
        "Linear": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "HGB_L1": lambda: HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
        "HGB_L2": lambda: HistGradientBoostingRegressor(
            loss="squared_error", max_depth=3, learning_rate=0.05, max_iter=300,
            min_samples_leaf=20, l2_regularization=1.0, random_state=SEED),
        "RF": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=8, max_features=0.5, random_state=SEED, n_jobs=-1),
    }


def scores(y, p):
    err = np.abs(p - y)
    sev_true = [severity_from_mm(v) for v in y]
    sev_pred = [severity_from_mm(v) for v in p]
    return {
        "mae": float(err.mean()),
        "within10": float((err <= 10).mean()),
        "within25": float((err <= 25).mean()),
        "sev_acc": float(np.mean([a == b for a, b in zip(sev_true, sev_pred)])),
        "sev_f1": float(f1_score(sev_true, sev_pred, average="macro")),
    }


# ── Decision policy ──────────────────────────────────────────────────────────
# The regressor predicts millimetres; severity is a decision on top of it. An
# absolute-error model pulls predictions toward the median, so banding its output
# at the label cut-offs (25 / 50 mm) finds almost no Deep potholes. The project's
# stated policy (MASTER_DOC, semantic calibration) is that hiding a hazard costs
# three times a wasted inspection, so the cut-offs on PREDICTED mm are chosen to
# minimise that cost, and are scored by cross-fitting: chosen on half the capture sessions,
# evaluated on the other half, and swapped.
COST_UNDER, COST_OVER = 3.0, 1.0
BAND_INDEX = {"Shallow": 0, "Moderate": 1, "Deep": 2}
BAND_NAMES = ["Shallow", "Moderate", "Deep"]


def band(pred, t_mod, t_deep):
    return np.where(pred < t_mod, 0, np.where(pred < t_deep, 1, 2))


def policy_cost(y, yhat):
    diff = yhat - y
    return float(np.sum(np.where(diff < 0, -COST_UNDER * diff, COST_OVER * diff)))


def choose_thresholds(y, pred):
    best = None
    for t1 in np.arange(5.0, 45.0, 0.5):
        for t2 in np.arange(t1 + 5.0, 70.0, 0.5):
            c = policy_cost(y, band(pred, t1, t2))
            if best is None or c < best[0]:
                best = (c, float(t1), float(t2))
    return best[1], best[2]


def policy_eval(rows):
    """Cross-fitted policy scores on one mask source's out-of-fold predictions."""
    y = rows.gt_severity.map(BAND_INDEX).values
    pred = rows.pred_mm.values
    groups = np.array(sorted(rows.group.unique()))
    half = set(np.random.default_rng(SEED).permutation(groups)[: len(groups) // 2])
    in_a = rows.group.isin(half).values
    yhat = np.empty_like(y)
    for fit_on, score_on in ((in_a, ~in_a), (~in_a, in_a)):
        t1, t2 = choose_thresholds(y[fit_on], pred[fit_on])
        yhat[score_on] = band(pred[score_on], t1, t2)
    point = band(pred, 25.0, 50.0)
    return {
        "crossfit": {
            "cost_per_pothole": policy_cost(y, yhat) / len(y),
            "accuracy": float((yhat == y).mean()),
            "macro_f1": float(f1_score(y, yhat, average="macro")),
            "under_reported": int((yhat < y).sum()),
            "deep_recall": float(((yhat == 2) & (y == 2)).sum() / max((y == 2).sum(), 1)),
        },
        "point_bands_25_50": {
            "cost_per_pothole": policy_cost(y, point) / len(y),
            "accuracy": float((point == y).mean()),
            "under_reported": int((point < y).sum()),
            "deep_recall": float(((point == 2) & (y == 2)).sum() / max((y == 2).sum(), 1)),
        },
        "n": int(len(y)),
    }, yhat


def summarise(per_fold):
    keys = per_fold[0].keys()
    return {k: {"mean": float(np.mean([s[k] for s in per_fold])),
                "sd": float(np.std([s[k] for s in per_fold]))} for k in keys}


def main():
    d = pd.read_csv(DATA)
    d["group"] = d.key.map(capture_sessions(d.key))
    gt = d[d.mask_source == "gt"]
    print(f"{len(gt)} measured potholes from {gt.key.nunique()} frames; "
          f"yolo rows {int((d.mask_source == 'yolo').sum())}, sam2 rows {int((d.mask_source == 'sam2').sum())}")

    # The old 39-feature MoGe/DA vectors, joined on the same potholes, for a like-for-like comparison.
    old = pd.read_csv(OLD) if os.path.exists(OLD) else None
    old_moge = [c for c in (old.columns if old is not None else []) if c.startswith("moge_")]

    variants = {
        "p90_affine": ["mf_p90_mm"],
        "ratio_affine": ["mf_p90_over_dist"],
        "metric": FEATURE_COLS,
        "metric_shape": FEATURE_COLS + SHAPE_COLS,
    }

    results = {}
    oof = {}          # out-of-fold residuals of each config, for error bars
    oof_rows = {}     # out-of-fold predictions, all repeats (each row once per repeat)

    def run(name, cols, model_name, make, train_on, frame=d):
        per_src = {s: [] for s in SOURCES}
        base = {"median": {s: [] for s in SOURCES}, "log_area": {s: [] for s in SOURCES}}
        resid = []
        preds = []
        for r_idx, _f, test in folds(gt.group.values):
            tr = frame[~frame.group.isin(test) & frame.mask_source.isin(train_on)]
            m = make()
            m.fit(tr[cols].values, tr.gt_mm.values)
            med = float(np.median(tr.gt_mm))
            la = LinearRegression().fit(tr[["sh_log_area_px"]].values, tr.gt_mm.values)
            for s in SOURCES:
                te = frame[frame.group.isin(test) & (frame.mask_source == s)]
                if len(te) < 5:
                    continue
                p = m.predict(te[cols].values)
                per_src[s].append(scores(te.gt_mm.values, p))
                base["median"][s].append(scores(te.gt_mm.values, np.full(len(te), med)))
                base["log_area"][s].append(scores(te.gt_mm.values, la.predict(te[["sh_log_area_px"]].values)))
                if s == "yolo":
                    resid.extend(np.abs(p - te.gt_mm.values))
                preds.append(te[["key", "group", "pothole", "mask_source", "gt_mm", "gt_severity"]]
                             .assign(pred_mm=p, repeat=r_idx))
        key = f"{name}/{model_name}/train={'+'.join(train_on)}"
        results[key] = {s: summarise(v) for s, v in per_src.items() if v}
        oof[key] = np.array(resid)
        oof_rows[key] = pd.concat(preds) if preds else None
        if "baselines" not in results:
            results["baselines"] = {b: {s: summarise(v) for s, v in bs.items() if v} for b, bs in base.items()}

    for vname, cols in variants.items():
        for mname, make in models().items():
            if vname.endswith("affine") and mname != "Linear":
                continue
            for train_on in (("gt",), ("gt", "yolo", "sam2")):
                run(vname, cols, mname, make, train_on)

    if old is not None:
        join = gt.merge(old[["key", "pothole"] + old_moge], on=["key", "pothole"], how="inner")
        # the old features exist only for human-polygon masks
        for mname in ("HGB_L1", "RF"):
            run("old_pipeline_moge39", old_moge, mname, models()[mname], ("gt",), frame=join)

    # ── report ──
    def fmt(r, s):
        if s not in r:
            return "      —      "
        return f"{r[s]['mae']['mean']:5.2f} ± {r[s]['mae']['sd']:4.2f}"

    print(f"\nMAE in mm, mean ± sd over {K}x{REPEATS} grouped folds.  Columns = mask used AT TEST TIME.")
    print(f"{'variant / model / training masks':52s} {'human':>13s} {'yolo':>13s} {'sam2':>13s}  sev-F1(yolo)")
    for b in ("median", "log_area"):
        r = results["baselines"][b]
        print(f"{'BASELINE ' + b:52s} {fmt(r,'gt'):>13s} {fmt(r,'yolo'):>13s} {fmt(r,'sam2'):>13s}  "
              f"{r['yolo']['sev_f1']['mean']:.3f}")
    ranked = sorted((k for k in results if k != "baselines"),
                    key=lambda k: results[k].get("yolo", results[k]["gt"])["mae"]["mean"])
    for k in ranked:
        r = results[k]
        f1 = r["yolo"]["sev_f1"]["mean"] if "yolo" in r else float("nan")
        print(f"{k:52s} {fmt(r,'gt'):>13s} {fmt(r,'yolo'):>13s} {fmt(r,'sam2'):>13s}  {f1:.3f}")

    # ── choose on the deployed condition: YOLO masks, cross-fitted safety cost ──
    def mean_oof(k):
        o = oof_rows[k]
        return (o.groupby(["key", "group", "pothole", "mask_source", "gt_mm", "gt_severity"], as_index=False)
                 .pred_mm.mean())

    policy = {}
    for k in ranked:
        if "yolo" not in results[k] or k.startswith("old_"):
            continue
        o = mean_oof(k)
        policy[k], _ = policy_eval(o[o.mask_source == "yolo"])
    print(f"\nSeverity policy on YOLO masks (under-reporting costs {COST_UNDER:.0f}x over-reporting), "
          "cut-offs cross-fitted by capture session:")
    print(f"{'config':52s} {'cost/pothole':>12s} {'acc':>6s} {'F1':>6s} {'under':>6s} {'Deep recall':>11s}")
    for k in sorted(policy, key=lambda k: policy[k]["crossfit"]["cost_per_pothole"])[:8]:
        c = policy[k]["crossfit"]
        print(f"{k:52s} {c['cost_per_pothole']:12.3f} {c['accuracy']:6.3f} {c['macro_f1']:6.3f} "
              f"{c['under_reported']:6d} {c['deep_recall']:11.3f}")
    best = min(policy, key=lambda k: policy[k]["crossfit"]["cost_per_pothole"])
    vname, mname, tr = best.split("/")
    train_on = tuple(tr.split("=")[1].split("+"))
    cols = variants[vname]
    best_oof = mean_oof(best)
    yo = best_oof[best_oof.mask_source == "yolo"]
    t_mod, t_deep = choose_thresholds(yo.gt_severity.map(BAND_INDEX).values, yo.pred_mm.values)
    _p, crossfit_bands = policy_eval(yo)
    best_oof["policy_band_crossfit"] = None
    best_oof.loc[yo.index, "policy_band_crossfit"] = [BAND_NAMES[i] for i in crossfit_bands]
    final = models()[mname]()
    all_rows = d[d.mask_source.isin(train_on)]
    final.fit(all_rows[cols].values, all_rows.gt_mm.values)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final, os.path.join(MODEL_DIR, "depth_regressor.joblib"))
    q80, q90 = (float(np.percentile(oof[best], q)) for q in (80, 90))
    meta = {
        "fitted": str(date.today()),
        "config": best,
        "features": cols,
        "target": "p90 bowl depth below the road plane, mm (PothRGBD RealSense)",
        "train_rows": int(len(all_rows)), "train_frames": int(all_rows.key.nunique()),
        "cv": results[best],
        "baselines": results["baselines"],
        "abs_error_q80_mm": q80, "abs_error_q90_mm": q90,
        # 1st-99th percentile of every feature in training. A served input outside
        # these ranges is extrapolation (e.g. a close-up photo MoGe places 7 m
        # away), and is flagged rather than silently trusted.
        "feature_ranges": {c: [float(np.percentile(all_rows[c], 1)), float(np.percentile(all_rows[c], 99))]
                           for c in cols},
        "label_bands_mm": {"Shallow": [None, 25], "Moderate": [25, 50], "Deep": [50, None]},
        "decision_thresholds_mm": {"moderate_from": t_mod, "deep_from": t_deep,
                                   "cost_under": COST_UNDER, "cost_over": COST_OVER},
        "policy_yolo": policy[best],
        "caveats": [
            "Single camera (RealSense D415), single operator, close range, Turkey.",
            "MoGe global scale varies per frame; the model partly learns PothRGBD's fixed camera height.",
            "Error-bar quantiles are out-of-fold absolute errors on YOLO masks.",
        ],
    }
    with open(os.path.join(MODEL_DIR, "depth_regressor_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    best_oof.to_csv(os.path.join(os.path.dirname(RESULTS), "metric_oof_predictions.csv"), index=False)

    print(f"\nChosen by cross-fitted safety cost on YOLO masks: {best}")
    print(f"  served cut-offs on predicted mm: Moderate from {t_mod:.1f}, Deep from {t_deep:.1f}")
    print(f"  80% of YOLO-mask predictions within {q80:.1f} mm, 90% within {q90:.1f} mm (out of fold)")
    print(f"  wrote {os.path.relpath(MODEL_DIR, PROJECT_DIR)}/depth_regressor.joblib (+ _meta.json)")


if __name__ == "__main__":
    main()
