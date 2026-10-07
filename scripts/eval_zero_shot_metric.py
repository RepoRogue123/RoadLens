"""
The published "segment, then read a metric depth model" approach, scored on our measured potholes.

arXiv 2505.21049 (and several others) pair a pothole segmenter with Depth-Anything-V2's metric
model trained for outdoor driving scenes (Virtual KITTI, depth up to 80 m), used zero-shot.
Here that model reads each held-out pothole with exactly our protocol (ring of road around the
outline, plane, 90th percentile of depth below it), so the only thing that differs from the
served model is where the depth map comes from.

Scored on the 207 PothRGBD potholes in the held-out sessions and the 79 Fan stereo potholes,
human outlines, against measured depth.

Usage:
    python scripts/eval_zero_shot_metric.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, "Depth-Anything-V2", "metric_depth"))   # the metric variant first
sys.path.insert(1, PROJECT_DIR)

from depth_anything_v2.dpt import DepthAnythingV2                               # noqa: E402  (metric_depth's)
from metric_features import road_plane_deviation                                # noqa: E402
from scripts.make_counterexamples import frame                                  # noqa: E402

CKPT = os.path.join(PROJECT_DIR, "Depth-Anything-V2", "checkpoints", "depth_anything_v2_metric_vkitti_vits.pth")
OUT = os.path.join(PROJECT_DIR, "ml_results", "comparison")


def bowl_mm(depth_mm, mask):
    res = road_plane_deviation(depth_mm, mask, np.isfinite(depth_mm))
    return None if res is None else float(np.percentile(res[0][mask > 0], 90))


def score(name, rows):
    d = pd.DataFrame(rows).dropna(subset=["pred"])
    e = d["pred"] - d["gt"]
    out = {"n": int(len(d)), "mae": float(e.abs().mean()), "bias": float(e.mean()),
           "r": float(np.corrcoef(d["pred"], d["gt"])[0, 1]), "within10": float((e.abs() <= 10).mean())}
    print(f"{name:28s} n={out['n']:3d}  MAE {out['mae']:6.2f} mm  bias {out['bias']:+7.2f}  r {out['r']:.3f}  within 10 mm {out['within10']:.1%}")
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=80.0)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model = model.to(dev).eval()
    infer = lambda bgr: model.infer_image(bgr, 518).astype(np.float64) * 1000.0     # metres -> mm   # noqa: E731

    t = pd.read_csv(os.path.join(PROJECT_DIR, "ml_results", "figures", "counterexamples", "test_potholes_all_methods.csv"))
    rows = []
    for key, g in t.groupby("key", sort=False):
        bgr, dep, polys = frame(key)
        d = infer(bgr)
        for r in g.itertuples():
            m = polys[int(r.pothole) - 1].astype(np.uint8)
            rows.append({"key": key, "pothole": r.pothole, "gt": r.gt_mm, "pred": bowl_mm(d, m)})
    res = {"pothrgbd_test": score("PothRGBD test (207)", rows)}

    from scripts.build_relief_targets import CACHE_DIR
    from scripts.depth_sources import native_bgr
    f = pd.read_csv(os.path.join(PROJECT_DIR, "ml_results", "depth_sources", "fan_potholes.csv"))
    frows = []
    for key, g in f.groupby("key"):
        z = np.load(os.path.join(CACHE_DIR, "fan", key + ".npz"))
        comp = cv2.connectedComponents(z["mask"])[1]
        d = infer(z["bgr"])
        for r in g.itertuples():
            frows.append({"key": key, "pothole": r.pothole, "gt": r.gt_mm,
                          "pred": bowl_mm(d, (comp == int(r.pothole)).astype(np.uint8))})
    res["fan"] = score("Fan stereo (79)", frows)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "zero_shot_dav2_metric.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote ml_results/comparison/zero_shot_dav2_metric.json")


if __name__ == "__main__":
    main()
