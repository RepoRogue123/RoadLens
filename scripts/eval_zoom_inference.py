"""
Does reading a small pothole from a close-up cut help, and does it hurt anywhere?

relief_depth.bowl_depth_zoomed cuts a window around any pothole that covers less than 5%
of the frame, so that it fills about 20% (the median of the measured photos), and reads the
depth there. This script compares that with the plain whole-frame reading, on human
outlines, for:

    PothRGBD validation sessions   where the decision is made
    PothRGBD test sessions         reported, not used to decide
    Fan stereo sets                a second camera, used as a development set

Rule, fixed beforehand: use the zoomed reading if, on PothRGBD VALIDATION potholes that
qualify for a zoom, it is no more than 0.5 mm worse than the plain reading, and on the
Fan sets it is better. PothRGBD has few small potholes, so the first condition is a
do-no-harm check; the second is where the evidence is.

Usage:
    python scripts/eval_zoom_inference.py [--weights a.pth,b.pth] [--tag served]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from scripts.build_relief_targets import CACHE_DIR, OUT_DIR                     # noqa: E402
from scripts.depth_sources import native_bgr                                    # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, load_polygons, timestamp_key  # noqa: E402
from scripts.train_relief_depth import split_keys                               # noqa: E402

TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")


def pothrgbd_items(split):
    d = pd.read_csv(TRAINSET)
    d = d[(d.mask_source == "gt") & d.key.isin(set(split_keys(split)))]
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    for key, grp in d.groupby("key"):
        bgr = cv2.imread(imgs[key])
        h, w = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0], mmap_mode="r").shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[key], h, w)
        yield bgr, bgr, [(polys[int(r.pothole) - 1], r.gt_mm, "pothrgbd") for r in grp.itertuples()]


def source_items(source):
    d = pd.read_csv(os.path.join(OUT_DIR, f"{source}_potholes.csv"))
    for key, grp in d.groupby("key"):
        z = np.load(os.path.join(CACHE_DIR, source, key + ".npz"))
        comp = cv2.connectedComponents(z["mask"])[1]
        yield z["bgr"], native_bgr(key), [((comp == int(r.pothole)).astype(np.uint8), r.gt_mm, r.group) for r in grp.itertuples()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=",".join(R.installed_weights()))
    ap.add_argument("--tag", default="served")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = [R.build_model(w).to(device).eval() for w in args.weights.split(",")]

    rows = []
    sets = {"pothrgbd val": pothrgbd_items("val"), "pothrgbd test": pothrgbd_items("test_pothrgbd"), "fan": source_items("fan")}
    for name, items in sets.items():
        for bgr, native, potholes in items:
            relief = R.predict_relief(bgr, nets)
            for mask, gt, group in potholes:
                plain = R.bowl_depth_mm(relief, mask)
                if plain is None:
                    continue
                zoomed = R.bowl_depth_zoomed(native, mask, nets)
                rows.append({"set": name, "group": group, "gt_mm": gt, "share": float(mask.mean()), "plain": plain,
                             "zoomed": np.nan if zoomed is None else zoomed})
        print(f"  {name} done", flush=True)
    d = pd.DataFrame(rows)
    d["served"] = d.zoomed.fillna(d.plain)                    # zoom where it applies, plain elsewhere

    results = {}
    print(f"\n{'set':14s} {'subset':22s} {'n':>4s} | {'plain MAE':>10s} {'bias':>7s} | {'zoomed MAE':>11s} {'bias':>7s}")
    for name in sets:
        s = d[d.set == name]
        parts = {"all": s, "zoom applies": s[s.zoomed.notna()], "no zoom": s[s.zoomed.isna()]}
        if name == "fan":
            parts.update({g: s[s.group == g] for g in sorted(s.group.unique())})
        for sub, t in parts.items():
            if not len(t):
                continue
            r = {"n": int(len(t)), "plain_mae": float((t.plain - t.gt_mm).abs().mean()), "plain_bias": float((t.plain - t.gt_mm).mean()),
                 "zoomed_mae": float((t.served - t.gt_mm).abs().mean()), "zoomed_bias": float((t.served - t.gt_mm).mean())}
            results[f"{name}|{sub}"] = r
            print(f"{name:14s} {sub:22s} {r['n']:4d} | {r['plain_mae']:10.2f} {r['plain_bias']:+7.1f} | {r['zoomed_mae']:11.2f} {r['zoomed_bias']:+7.1f}")

    v, f = results.get("pothrgbd val|zoom applies"), results["fan|all"]
    harm = (v["zoomed_mae"] - v["plain_mae"]) if v else 0.0
    ok = harm <= 0.5 and f["zoomed_mae"] < f["plain_mae"]
    results["decision"] = {"validation_change_mm": harm, "fan_change_mm": f["zoomed_mae"] - f["plain_mae"], "use_zoom": bool(ok)}
    print(f"\nrule: validation change {harm:+.2f} mm (limit +0.5), Fan change {f['zoomed_mae'] - f['plain_mae']:+.2f} mm -> "
          f"{'USE the zoomed reading' if ok else 'do NOT use the zoomed reading'}")
    d.to_csv(os.path.join(OUT_DIR, f"zoom_eval_{args.tag}.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"zoom_eval_{args.tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote ml_results/depth_sources/zoom_eval_{args.tag}.json")


if __name__ == "__main__":
    main()
