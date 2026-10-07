"""
Is the fine-tuned relief network a better depth source than MoGe + regressor?

Scored on the PothRGBD capture sessions held out from BOTH the relief network and the
segmenter (archive/yolo_seg_v2/test_pothrgbd), against RealSense labels, with the human
outline and with the production segmenter's outline. A pothole the segmenter misses
counts against every method (no estimate = under-reported).

    median        always predict the training median
    regressor     the served model family (MoGe metric features -> HGB), refitted on the
                  training sessions only
    relief        the relief network's own reading: label protocol on its output, no fitting
    relief+cal    relief with a straight-line calibration fitted on the validation sessions
    blend         a straight-line blend of regressor and relief, fitted on the validation sessions

Severity cut-offs for every method are chosen on the validation sessions with the
project's 3:1 cost and applied unchanged to the test sessions.

Usage:
    python scripts/eval_relief_depth.py
"""
import argparse
import glob
import json
import os
import shutil
import sys
from datetime import date

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import metric_severity                                                          # noqa: E402
import relief_depth as R                                                        # noqa: E402
import segmentation                                                             # noqa: E402
from metric_features import extract_metric_features, shape_features             # noqa: E402
from scripts.build_metric_trainset import iou, moge_for                         # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, load_polygons, timestamp_key  # noqa: E402
from scripts.train_metric_regressor import (                                    # noqa: E402
    BAND_INDEX, band, choose_thresholds, models, policy_cost,
)
from scripts.train_relief_depth import split_keys                               # noqa: E402

TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
OUT_CSV = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "relief_eval_potholes.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "relief_eval.json")
METHODS = ["median", "regressor", "relief", "relief+cal", "blend"]


def collect(keys, split, d, reg, cols, net):
    """One row per measured pothole per mask source, with both models' readings."""
    import moge_backend
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    rows = []
    gt = d[d.key.isin(keys) & (d.mask_source == "gt")]
    for key, grp in gt.groupby("key"):
        bgr = cv2.imread(imgs[key])
        h, w = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0]).shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[key], h, w)
        preds = segmentation._extract_binary_masks(bgr, segmentation.MODEL)
        g = moge_for(key, bgr, moge_backend)
        relief = R.predict_relief(bgr, net)                 # one network, or an average of several
        for _, r in grp.iterrows():
            gm = polys[int(r.pothole) - 1]
            ious = [iou(p, gm) for p in preds]
            best = int(np.argmax(ious)) if ious else -1
            for src, mask in (("human", gm), ("yolo", preds[best] if ious and ious[best] >= 0.1 else None)):
                rec = {"split": split, "key": key, "pothole": int(r.pothole), "mask": src,
                       "gt_mm": r.gt_mm, "gt_band": BAND_INDEX[r.gt_severity],
                       "reg_mm": np.nan, "relief_mm": np.nan}
                if mask is not None:
                    mf = extract_metric_features(mask, g["depth"], g["intrinsics"], normal=g["normal"], valid=g["mask"])
                    sf = shape_features(mask)
                    if mf and sf:
                        f = {**mf, **sf}
                        rec["reg_mm"] = float(np.clip(reg.predict(np.array([[f[c] for c in cols]]))[0], 0, 250))
                    b = R.bowl_depth_mm(relief, mask)
                    if b is not None:
                        rec["relief_mm"] = b
                rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=",".join(R.installed_weights()),
                    help="relief network checkpoint to score; several, comma-separated, are averaged")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--install", action="store_true",
                    help="serve these weights: copy to ml_models/metric/ and write relief_meta.json")
    args = ap.parse_args()
    import torch
    net = [R.build_model(w).to("cuda" if torch.cuda.is_available() else "cpu").eval()
           for w in args.weights.split(",")]
    print(f"relief network: {args.weights}")
    d = pd.read_csv(TRAINSET)
    train_k, val_k, test_k = (set(split_keys(s)) for s in ("train", "val", "test_pothrgbd"))
    _m, meta = metric_severity._load()
    cols, mname = meta["features"], meta["config"].split("/")[1]
    train_on = meta["config"].split("=")[1].split("+")
    tr = d[d.key.isin(train_k) & d.mask_source.isin(train_on)]
    reg = models()[mname]()
    reg.fit(tr[cols].values, tr.gt_mm.values)
    median = float(np.median(d[d.key.isin(train_k) & (d.mask_source == "gt")].gt_mm))

    t = pd.DataFrame(collect(val_k, "val", d, reg, cols, net) + collect(test_k, "test", d, reg, cols, net))
    t["median"] = np.where(t.reg_mm.notna() | (t["mask"] == "human"), median, np.nan)
    t.loc[t.reg_mm.isna() & t.relief_mm.isna(), "median"] = np.nan         # missed by the segmenter
    t = t.rename(columns={"reg_mm": "regressor", "relief_mm": "relief"})

    results = {}
    print(f"\ntraining median {median:.1f} mm; regressor {mname} on {tr.key.nunique()} training-session frames")
    for src in ("human", "yolo"):
        v, te = t[(t.split == "val") & (t["mask"] == src)].copy(), t[(t.split == "test") & (t["mask"] == src)].copy()
        vf = v.dropna(subset=["regressor", "relief"])
        cal = LinearRegression().fit(vf[["relief"]].values, vf.gt_mm.values)
        blend = LinearRegression().fit(vf[["regressor", "relief"]].values, vf.gt_mm.values)
        for frame in (v, te):
            ok = frame.regressor.notna() & frame.relief.notna()
            frame["relief+cal"], frame["blend"] = np.nan, np.nan
            frame.loc[ok, "relief+cal"] = cal.predict(frame.loc[ok, ["relief"]].values)
            frame.loc[ok, "blend"] = blend.predict(frame.loc[ok, ["regressor", "relief"]].values)
        print(f"\n== {src} masks: {len(te)} measured potholes in {te.key.nunique()} held-out frames "
              f"({int((te.gt_band == 2).sum())} Deep); found {te.relief.notna().mean():.1%}")
        print(f"   calibration fitted on {len(vf)} validation potholes: relief+cal = {cal.coef_[0]:.2f} x relief + {cal.intercept_:.1f};"
              f"  blend = {blend.coef_[0]:.2f} x regressor + {blend.coef_[1]:.2f} x relief + {blend.intercept_:.1f}")
        print(f"   {'method':11s} {'MAE mm':>7s} {'r':>6s} {'within 10mm':>12s} | {'cut-offs':>11s} {'cost':>6s} {'correct':>8s} {'under/missed':>13s} {'Deep found':>11s}")
        results[src] = {}
        for m in METHODS:
            vv = v.dropna(subset=[m])
            t1, t2 = choose_thresholds(vv.gt_band.values, vv[m].values)
            found = te[m].notna()
            p, g = te.loc[found, m].values, te.loc[found, "gt_mm"].values
            yhat = np.full(len(te), -1)
            yhat[found.values] = band(p, t1, t2)
            y = te.gt_band.values
            res = {"mae": float(np.abs(p - g).mean()),
                   "r": float(np.corrcoef(p, g)[0, 1]) if np.std(p) > 0 else float("nan"),
                   "within10": float((np.abs(p - g) <= 10).mean()),
                   "cutoffs": [t1, t2],
                   "cost": policy_cost(y, np.where(yhat < 0, 0, yhat)) / len(y),     # missed = reported nothing
                   "correct": float((yhat == y).mean()),
                   "under_or_missed": int((yhat < y).sum()),
                   "deep_found": int(((yhat == 2) & (y == 2)).sum()), "n_deep": int((y == 2).sum()), "n": int(len(y))}
            results[src][m] = res
            print(f"   {m:11s} {res['mae']:7.2f} {res['r']:6.3f} {res['within10']:12.1%} | {t1:5.1f}/{t2:5.1f} {res['cost']:6.3f} "
                  f"{res['correct']:8.1%} {res['under_or_missed']:13d} {res['deep_found']:>8d}/{res['n_deep']}")

    suffix = f"_{args.tag}" if args.tag else ""
    t.to_csv(OUT_CSV.replace(".csv", suffix + ".csv"), index=False)
    out_json = OUT_JSON.replace(".json", suffix + ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(out_json, PROJECT_DIR)}")

    if args.install:
        # Served cut-offs use every pothole the networks never trained on (validation and
        # test sessions, segmenter outlines). The honest estimate of how they perform is
        # the row above, where they were chosen on validation alone and scored on test.
        held = t[(t["mask"] == "yolo")].dropna(subset=["relief"])
        t_mod, t_deep = choose_thresholds(held.gt_band.values, held.relief.values)
        te = held[held.split == "test"]
        err = np.abs(te.relief - te.gt_mm)
        for old in R.installed_weights():
            os.remove(old)
        for i, w in enumerate(args.weights.split(","), start=1):
            shutil.copy2(w, os.path.join(R.MODEL_DIR, f"relief_vits_{i}.pth"))
        meta = {
            "fitted": str(date.today()),
            "weights_from": args.weights.split(","),
            "target": "p90 relief below the ring-fitted road plane, mm (PothRGBD RealSense)",
            "decision_thresholds_mm": {"moderate_from": t_mod, "deep_from": t_deep,
                                       "chosen_on": f"{len(held)} held-out potholes (val + test sessions, segmenter outlines)"},
            "abs_error_q80_mm": float(np.percentile(err, 80)),
            "abs_error_q90_mm": float(np.percentile(err, 90)),
            "test_yolo": results["yolo"]["relief"], "test_human": results["human"]["relief"],
            "baseline_regressor_test_yolo": results["yolo"]["regressor"],
            "caveats": [
                "Trained and tested on one camera (RealSense D415), one operator, close range, looking down.",
                "No measured depth exists for any other kind of photo; behaviour there is unverified.",
            ],
        }
        with open(R.META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"installed {len(args.weights.split(','))} network(s) -> ml_models/metric/relief_vits_*.pth; "
              f"cut-offs Moderate from {t_mod}, Deep from {t_deep}; 80% of test errors within {meta['abs_error_q80_mm']:.1f} mm")


if __name__ == "__main__":
    main()
