"""
Head-to-head on measured labels: the severity RoadLens serves today versus the
metric depth model, on the same YOLO-detected PothRGBD potholes.

    legacy_vote      rule classifier + 4 pseudo-label models, severity-biased majority
    legacy_served    legacy_vote after the DINOv2 semantic override (what /analyze returns)
    metric+override  the metric verdict passed through the same DINOv2 override
    metric_point     bowl depth predicted by the metric model, banded at the label cut-offs
                     (25 / 50 mm), from OUT-OF-FOLD predictions (no model saw its frame)
    metric_policy    the same predictions banded at the served, cost-sensitive cut-offs,
                     which were cross-fitted by frame (chosen on one half, scored on the other)

All three are scored against the RealSense label under the human polygon. The
legacy path runs through api.py's own functions (majority_vote, semantic_override,
extract_ml_features), so it is the served logic, not a re-implementation.

Run after build_metric_trainset.py and train_metric_regressor.py:
    python scripts/compare_served_severity.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import api                                                          # noqa: E402
import segmentation                                                 # noqa: E402
from scripts.build_metric_trainset import iou                       # noqa: E402
from scripts.pothrgbd_metric_labels import (  # noqa: E402
    DATA_DIR, load_polygons, severity_from_mm, timestamp_key,
)

OUT = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
BANDS = ["Shallow", "Moderate", "Deep"]


def legacy_for(bgr, mask, depth_map, scaler, models, expected):
    feats = api.extract_depth_features(mask, depth_map)
    rule = api.normalize_severity(api.classify_severity(feats))
    votes = {"Rule-Based": rule}
    vec = api.extract_ml_features(mask, depth_map, expected)
    if vec is not None and scaler is not None:
        x = scaler.transform(vec)
        for name, m in models.items():
            votes[name] = api.normalize_severity(api.SEVERITY_MAP.get(int(m.predict(x)[0]), "Unknown"))
    consensus, _c, _t = api.majority_vote(list(votes.values()))
    served, verdict, ratio, n = consensus, "unavailable", None, 0
    if api._HAS_GEOMETRY:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        geo = api.extract_all_geometry_features(mask, depth_map, rgb) or {}
        ratio = geo.get(api.semantic_config.RATIO_KEY)
        n = int(geo.get("dinov2_patch_count", 0) or 0)
        served, verdict, _ill = api.semantic_override(consensus, rule, ratio, n)
    return consensus, served, verdict, rule, ratio, n


def report(name, y, p):
    acc = float(np.mean(np.array(y) == np.array(p)))
    f1 = float(f1_score(y, p, labels=BANDS, average="macro"))
    cm = confusion_matrix(y, p, labels=BANDS)
    # safety error: truly Deep/Moderate reported milder
    under = int(sum(cm[i, j] for i in range(3) for j in range(3) if j < i))
    # The project's policy cost: each band of under-reporting costs 3, of over-reporting 1.
    # One number that cannot be gamed by calling everything Deep.
    cost = float(sum(cm[i, j] * (3 * (i - j) if j < i else j - i)
                     for i in range(3) for j in range(3)) / len(y))
    print(f"\n{name}\n  accuracy {acc:.3f}   macro-F1 {f1:.3f}   "
          f"under-reported (reported milder than measured) {under}/{len(y)}   policy cost/pothole {cost:.3f}")
    print("            pred  " + "  ".join(f"{b[:5]:>6s}" for b in BANDS))
    for i, b in enumerate(BANDS):
        print(f"  true {b:9s}   " + "  ".join(f"{v:6d}" for v in cm[i]))
    return {"accuracy": acc, "macro_f1": f1, "under_reported": under, "policy_cost": cost, "n": len(y),
            "confusion": cm.tolist()}


def main():
    oof = pd.read_csv(os.path.join(OUT, "metric_oof_predictions.csv"))
    oof = oof[oof.mask_source == "yolo"].copy()
    print(f"{len(oof)} YOLO-detected measured potholes with out-of-fold metric predictions")

    scaler, models = api.load_ml_artifacts()
    expected = int(getattr(scaler, "n_features_in_", 11))
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}

    # The legacy pass does not depend on the metric model, so it is cached and
    # only recomputed when the set of potholes changes.
    cache = os.path.join(OUT, "legacy_served_on_pothrgbd.csv")
    if os.path.exists(cache):
        leg = pd.read_csv(cache)
        if set(zip(leg.key, leg.pothole)) >= set(zip(oof.key, oof.pothole)):
            finish(oof.merge(leg, on=["key", "pothole"], how="left"))
            return

    legacy, served, verdicts, rules, ratios, patches = [], [], [], [], [], []
    for n, (key, grp) in enumerate(oof.groupby("key"), start=1):
        bgr = cv2.imread(imgs[key])
        h, w = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0]).shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[key], h, w)
        yolo = segmentation._extract_binary_masks(bgr, segmentation.MODEL)
        depth = api.get_depth_map(bgr)
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h))
        for _, row in grp.iterrows():
            gm = polys[int(row.pothole) - 1]
            ym = yolo[int(np.argmax([iou(m, gm) for m in yolo]))]
            c, s, v, rule, ratio, n_p = legacy_for(bgr, ym, depth, scaler, models, expected)
            legacy.append(c)
            served.append(s)
            verdicts.append(v)
            rules.append(rule)
            ratios.append(ratio)
            patches.append(n_p)
        if n % 100 == 0:
            print(f"  {n} frames", flush=True)

    order = oof.sort_values("key")
    leg = pd.DataFrame({"key": [k for k, g in oof.groupby("key") for _ in range(len(g))],
                        "pothole": [p for _k, g in oof.groupby("key") for p in g.pothole],
                        "legacy_vote": legacy, "legacy_served": served, "semantic_verdict": verdicts,
                        "rule": rules, "dinov2_ratio": ratios, "dinov2_patches": patches})
    leg.to_csv(cache, index=False)
    finish(order.merge(leg, on=["key", "pothole"], how="left"))


def finish(df):
    df = df.copy()
    df["metric_point"] = [severity_from_mm(v) for v in df.pred_mm]
    df["metric_policy"] = df["policy_band_crossfit"]
    df["metric_policy_plus_override"] = [
        api.semantic_override(m, m, None if pd.isna(r) else r, int(n))[0]
        for m, r, n in zip(df.metric_policy, df.dinov2_ratio, df.dinov2_patches)]
    df.to_csv(os.path.join(OUT, "served_severity_comparison.csv"), index=False)

    y = df.gt_severity.tolist()
    res = {
        "legacy_vote": report("LEGACY VOTE (rule + 4 pseudo-label models)", y, df.legacy_vote.tolist()),
        "legacy_served": report("LEGACY SERVED (vote + DINOv2 override) — what /analyze returned before", y,
                                df.legacy_served.tolist()),
        "metric_point": report("METRIC MODEL, label cut-offs 25/50 mm", y, df.metric_point.tolist()),
        "metric_policy": report("METRIC MODEL, served cost-sensitive cut-offs (cross-fitted)", y,
                                df.metric_policy.tolist()),
        "metric_policy_plus_override": report("METRIC MODEL (policy) + DINOv2 override", y,
                                              df.metric_policy_plus_override.tolist()),
        "majority_class": report("BASELINE: always the most common band", y, [pd.Series(y).mode()[0]] * len(y)),
        "semantic_verdicts": df.semantic_verdict.value_counts().to_dict(),
    }
    for k, r in res.items():
        if isinstance(r, dict) and "confusion" in r:
            cm = np.array(r["confusion"])
            r["deep_recall"] = float(cm[2, 2] / max(cm[2].sum(), 1))
    with open(os.path.join(OUT, "served_severity_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\n  semantic verdicts: {res['semantic_verdicts']}")
    print("  wrote ml_results/pothrgbd/served_severity_comparison.(csv|json)")


if __name__ == "__main__":
    main()
