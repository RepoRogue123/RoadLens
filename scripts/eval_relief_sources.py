"""
Score the relief network in millimetres on a measured-depth dataset other than PothRGBD.

Reads the frames and pothole labels written by scripts/build_relief_targets.py, runs the
network, and applies the label protocol to its prediction, with the dataset's own outline
and with the production segmenter's outline. A pothole the segmenter misses counts as
under-reported.

    median    always predict the PothRGBD training median (what "knowing nothing" scores)
    relief    the network

Severity uses the served cut-offs unchanged (ml_models/metric/relief_meta.json): this is
what the app would have answered. Nothing here is fitted on the dataset being scored.

Usage:
    python scripts/eval_relief_sources.py --source fan
    python scripts/eval_relief_sources.py --source fan --weights a.pth,b.pth --tag v4
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import metric_severity                                                          # noqa: E402
import relief_depth as R                                                        # noqa: E402
import segmentation                                                             # noqa: E402
from scripts.build_metric_trainset import iou                                   # noqa: E402
from scripts.build_relief_targets import CACHE_DIR, OUT_DIR                     # noqa: E402
from scripts.train_metric_regressor import policy_cost                          # noqa: E402
from scripts.train_relief_depth import split_keys                               # noqa: E402

TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
BANDS = {"Shallow": 0, "Moderate": 1, "Deep": 2}


def band_of(mm):
    return BANDS[metric_severity.severity_from_mm(mm)]


def dense_error(nets, source, keys):
    """
    Per-pixel relief error over every measured pixel of each frame, after removing the
    best-fitting plane from the difference (a frame-wide tilt is not what is being asked).
    Returns per-frame median absolute error in mm.
    """
    out = []
    for k in keys:
        z = np.load(os.path.join(CACHE_DIR, source, k + ".npz"))
        v = z["valid"]
        if v.sum() < 500:
            continue
        diff = (R.predict_relief(z["bgr"], nets) - z["relief"]).astype(np.float64)
        ys, xs = np.nonzero(v)
        a = np.column_stack([xs, ys, np.ones(len(xs))])
        c, *_ = np.linalg.lstsq(a, diff[ys, xs], rcond=None)
        out.append(float(np.median(np.abs(diff[ys, xs] - a @ c))))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default=",".join(R.installed_weights()))
    ap.add_argument("--tag", default="served")
    ap.add_argument("--split", default="", help="RSRD: score only the drives of this role (test / val)")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = [R.build_model(w).to(device).eval() for w in args.weights.split(",")]
    meta = R.meta()
    labels = pd.read_csv(os.path.join(OUT_DIR, f"{args.source}_potholes.csv"))
    dense = None
    if args.split:
        from scripts.depth_sources import rsrd_split
        keys = sorted(f[:-4] for f in os.listdir(os.path.join(CACHE_DIR, args.source)) if f.endswith(".npz"))
        role = rsrd_split(keys)
        keys = [k for k in keys if role[k] == args.split]
        labels = labels[labels.key.isin(set(keys))]
        dense = dense_error(nets, args.source, keys)
        print(f"{args.source} {args.split} drives: {len(keys)} frames; per-pixel relief error, median over frames "
              f"{np.median(dense):.2f} mm (frames' medians: p25 {np.percentile(dense, 25):.2f}, p75 {np.percentile(dense, 75):.2f})")
    t = pd.read_csv(TRAINSET)
    median = float(np.median(t[t.key.isin(set(split_keys("train"))) & (t.mask_source == "gt")].gt_mm))

    rows = []
    for key, grp in labels.groupby("key"):
        z = np.load(os.path.join(CACHE_DIR, args.source, key + ".npz"))
        bgr, comp = z["bgr"], cv2.connectedComponents(z["mask"])[1]
        relief = R.predict_relief(bgr, nets)
        found = segmentation._extract_binary_masks(bgr, segmentation.MODEL)
        for _, r in grp.iterrows():
            human = (comp == int(r.pothole)).astype(np.uint8)
            ious = [iou(p, human) for p in found]
            best = int(np.argmax(ious)) if ious else -1
            for src, mask in (("human", human), ("segmenter", found[best] if ious and ious[best] >= 0.1 else None)):
                b = R.bowl_depth_mm(relief, mask) if mask is not None else None
                rows.append({**r.to_dict(), "mask": src, "relief": np.nan if b is None else b,
                             "median": np.nan if b is None else median})
    d = pd.DataFrame(rows)

    results = {"weights": args.weights.split(","), "training_median_mm": median,
               "cutoffs_mm": meta["decision_thresholds_mm"], "rows": {}, "split": args.split,
               "dense_median_mm": float(np.median(dense)) if dense is not None else None}
    print(f"\n== {args.source}: {labels.key.nunique()} frames, {len(labels)} measured potholes; "
          f"served cut-offs Moderate from {meta['decision_thresholds_mm']['moderate_from']:g}, Deep from {meta['decision_thresholds_mm']['deep_from']:g} mm")
    print(f"   {'subset':14s} {'mask':10s} {'method':7s} {'n':>3s} {'found':>6s} {'MAE mm':>7s} {'bias':>6s} {'r':>6s} {'<=10mm':>7s} "
          f"{'vs ruler':>9s} | {'cost':>5s} {'correct':>8s} {'under/missed':>13s}")
    subsets = {"all": d, "not set aside": d[~d.flagged]}
    subsets.update({g: d[d.group == g] for g in sorted(d.group.unique())})
    for name, sub in subsets.items():
        for src in ("human", "segmenter"):
            s = sub[sub["mask"] == src]
            if not len(s):
                continue
            y = np.array([band_of(v) for v in s.gt_mm])
            for m in ("median", "relief"):
                ok = s[m].notna().values
                p, g = s[m].values[ok], s.gt_mm.values[ok]
                yhat = np.full(len(s), -1)
                yhat[ok] = [BANDS[metric_severity.decide(v, meta)] for v in p]
                res = {"n": int(len(s)), "found": float(ok.mean()),
                       "mae": float(np.abs(p - g).mean()) if ok.any() else None,
                       "bias": float((p - g).mean()) if ok.any() else None,
                       "r": float(np.corrcoef(p, g)[0, 1]) if ok.sum() > 2 and np.std(p) > 0 else None,
                       "within10": float((np.abs(p - g) <= 10).mean()) if ok.any() else None,
                       "mae_vs_ruler": float(np.abs(p - s.perp_mm.values[ok]).mean()) if ok.any() and s.perp_mm.notna().any() else None,
                       "cost": float(policy_cost(y, np.where(yhat < 0, 0, yhat)) / len(y)),
                       "correct": float((yhat == y).mean()), "under_or_missed": int((yhat < y).sum())}
                results["rows"][f"{name}|{src}|{m}"] = res
                f = lambda v, spec: format(v, spec) if v is not None else "   -"          # noqa: E731
                print(f"   {name:14s} {src:10s} {m:7s} {res['n']:3d} {res['found']:6.1%} {f(res['mae'], '7.2f')} {f(res['bias'], '+6.1f')} "
                      f"{f(res['r'], '6.3f')} {f(res['within10'], '7.1%')} {f(res['mae_vs_ruler'], '9.2f')} | "
                      f"{res['cost']:5.3f} {res['correct']:8.1%} {res['under_or_missed']:13d}")

    name = f"{args.source}{'_' + args.split if args.split else ''}_relief_eval_{args.tag}"
    d.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)
    out = os.path.join(OUT_DIR, name + ".json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
