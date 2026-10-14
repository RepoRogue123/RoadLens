"""
Which single-photo model knows how far away the road is? Scored against measured distance.

A pothole's width in millimetres needs the camera-to-road distance: the photo gives its
ANGULAR size accurately (MoGe's angular size is within 25% of the truth for every Fan
pothole, scripts/eval_diameter_fan.py), but the distance is not in the pixels. MoGe's own
distance is off by a camera-dependent factor (2.6x on PothRGBD, 5.6x on Fan). This script
asks whether another metric depth model does better, on three cameras where the true
distance is measured:

    pothrgbd   RealSense, 0.4-1.65 m, looking down          (held-out test sessions)
    fan        stereo rig, about 0.4 m, looking down         (all 79 potholes)
    rsrd       vehicle stereo + LiDAR, road 2-9 m ahead      (held-out test drives, sampled)

Per pothole, the distance is the median depth over the ring of road around it (the ring the
label protocol uses), measured and predicted on the same pixels. Models:

    moge         MoGe-3 ViT-L (what the served regressor uses)
    dav2_metric  Depth-Anything-V2 metric, outdoor (Virtual KITTI), ViT-S
    depthpro     Apple Depth Pro (estimates its own focal length), via transformers

Reported per model and camera: median ratio predicted / true (the bias), the share within
25% and 50% of the truth, and the p10-p90 ratio spread. What matters for photos from an
unknown camera is the spread ACROSS cameras, so the median ratio of each camera is also
compared with the others.

Output: ml_results/depth_sources/camera_distance.csv and camera_distance.json

Usage:
    python scripts/eval_camera_distance.py [--models moge,dav2_metric,depthpro] [--rsrd-frames 120]
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

from metric_features import road_plane_deviation                                # noqa: E402
from scripts.build_relief_targets import CACHE_DIR, to_scale                     # noqa: E402
from scripts.depth_sources import PRESENT, fan_frames, rsrd_split                # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, DEPTH_VALID_MIN, load_polygons, timestamp_key  # noqa: E402
from scripts.train_relief_depth import split_keys                               # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "depth_sources")
SEED = 20261013


# ── models: each maps a BGR image to HxW z-depth in mm at the image's resolution ──
def load_models(names):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    if "moge" in names:
        import moge_backend

        def moge(bgr):
            g = moge_backend.get_geometry(bgr)
            d = np.asarray(g["depth"], np.float64) * 1000.0
            v = np.asarray(g["mask"]) > 0.5
            d[~v] = np.nan
            return cv2.resize(d, bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        models["moge"] = moge
    if "dav2_metric" in names:
        sys.path.insert(0, os.path.join(PROJECT_DIR, "Depth-Anything-V2", "metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2 as MetricDAV2
        sys.path.pop(0)
        m = MetricDAV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=80.0)
        m.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "Depth-Anything-V2", "checkpoints",
                                                  "depth_anything_v2_metric_vkitti_vits.pth"), map_location="cpu"))
        m = m.to(dev).eval()
        models["dav2_metric"] = lambda bgr: m.infer_image(bgr, 518).astype(np.float64) * 1000.0
    if "depthpro" in names:
        from PIL import Image
        from transformers import DepthProForDepthEstimation, DepthProImageProcessor
        proc = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")
        net = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch.float16).to(dev).eval()

        def depthpro(bgr):
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            x = proc(images=img, return_tensors="pt").to(dev, torch.float16)
            with torch.no_grad():
                out = net(**x)
            post = proc.post_process_depth_estimation(out, target_sizes=[(bgr.shape[0], bgr.shape[1])])
            return post[0]["predicted_depth"].float().cpu().numpy().astype(np.float64) * 1000.0
        models["depthpro"] = depthpro
    return models


# ── data: (source, key, bgr, [(mask, true depth map mm, valid)]) ──
def pothrgbd_items():
    keys = set(split_keys("test_pothrgbd"))
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    for k in sorted(keys):
        dp = glob.glob(os.path.join(DATA_DIR, "depths", f"{k}*.npy"))
        if k not in imgs or k not in labs or not dp:
            continue
        depth = np.load(dp[0]).astype(np.float64)
        bgr = cv2.imread(imgs[k])
        if bgr.shape[:2] != depth.shape:
            bgr = cv2.resize(bgr, depth.shape[::-1], interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[k], *depth.shape)
        yield "pothrgbd", k, bgr, depth, depth >= DEPTH_VALID_MIN, polys


def fan_items():
    for raw in fan_frames():
        f = to_scale(raw)
        n, comp = cv2.connectedComponents(f["mask"])
        yield "fan", f["key"], f["bgr"], f["depth_mm"].astype(np.float64), f["depth_mm"] > 0, \
            [(comp == c).astype(np.uint8) for c in range(1, n)]


def rsrd_items(n_frames, rng):
    keys = sorted(f[:-4] for f in os.listdir(os.path.join(CACHE_DIR, "rsrd")) if f.endswith(".npz"))
    role = rsrd_split(keys)
    test = [k for k in keys if role[k] == "test"]
    for k in sorted(rng.choice(test, min(n_frames, len(test)), replace=False)):
        z = np.load(os.path.join(CACHE_DIR, "rsrd", k + ".npz"))
        n, comp = cv2.connectedComponents(z["mask"])
        polys = [(comp == c).astype(np.uint8) for c in range(1, n)]
        if polys:
            d = z["depth_mm"].astype(np.float64)
            yield "rsrd", k, z["bgr"], d, d > 0, polys


def ring_median(depth_mm, mask, valid):
    res = road_plane_deviation(np.where(valid, depth_mm, 0.0), mask, valid & np.isfinite(depth_mm))
    if res is None:
        return None
    ring = res[1]
    v = depth_mm[ring & valid & np.isfinite(depth_mm)]
    return float(np.median(v)) if v.size >= 50 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="moge,dav2_metric,depthpro")
    ap.add_argument("--rsrd-frames", type=int, default=120)
    ap.add_argument("--tag", default="", help="suffix for the output files, so runs of different models can be merged")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    models = load_models(args.models.split(","))
    sources = [pothrgbd_items(), fan_items()]
    if PRESENT["rsrd"] and os.path.isdir(os.path.join(CACHE_DIR, "rsrd")):
        sources.append(rsrd_items(args.rsrd_frames, rng))

    rows = []
    for items in sources:
        n = 0
        for source, key, bgr, depth, valid, polys in items:
            preds = {name: fn(bgr) for name, fn in models.items()}
            for i, m in enumerate(polys, start=1):
                true = ring_median(depth, m, valid)
                if true is None:
                    continue
                row = {"source": source, "key": key, "pothole": i, "true_mm": true}
                for name, p in preds.items():
                    row[name] = ring_median(p, m, np.isfinite(p) & (p > 0))
                rows.append(row)
            n += 1
            if n % 25 == 0:
                print(f"  {source}: {n} frames", flush=True)
        print(f"{source} done", flush=True)

    d = pd.DataFrame(rows)
    res = {}
    print(f"\n{'model':12s} {'camera':9s} {'n':>4s} {'median ratio':>13s} {'p10-p90':>13s} {'within 25%':>11s} {'within 50%':>11s}")
    for name in models:
        res[name] = {}
        for src, g in d.groupby("source"):
            r = (g[name] / g.true_mm).dropna()
            s = {"n": int(len(r)), "median_ratio": float(r.median()),
                 "p10": float(r.quantile(0.1)), "p90": float(r.quantile(0.9)),
                 "within25": float((np.abs(r - 1) <= 0.25).mean()), "within50": float((np.abs(r - 1) <= 0.5).mean())}
            res[name][src] = s
            print(f"{name:12s} {src:9s} {s['n']:4d} {s['median_ratio']:13.2f} {s['p10']:6.2f}-{s['p90']:<6.2f} "
                  f"{s['within25']:11.0%} {s['within50']:11.0%}")
        allr = (d[name] / d.true_mm).dropna()
        res[name]["all_cameras"] = {"n": int(len(allr)), "median_ratio": float(allr.median()),
                                    "p10": float(allr.quantile(0.1)), "p90": float(allr.quantile(0.9)),
                                    "within25": float((np.abs(allr - 1) <= 0.25).mean()),
                                    "within50": float((np.abs(allr - 1) <= 0.5).mean())}
        a = res[name]["all_cameras"]
        print(f"{name:12s} {'ALL':9s} {a['n']:4d} {a['median_ratio']:13.2f} {a['p10']:6.2f}-{a['p90']:<6.2f} "
              f"{a['within25']:11.0%} {a['within50']:11.0%}")
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "camera_distance" + (f"_{args.tag}" if args.tag else "")
    d.to_csv(os.path.join(OUT_DIR, name + ".csv"), index=False)
    with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote ml_results/depth_sources/{name}.json")


if __name__ == "__main__":
    main()
