"""
Does the relief network see pothole SHAPE on a camera it was never trained on?

Pothole-600 is a different camera (a stereo rig, 400x400 crops) from PothRGBD's
RealSense. It carries no millimetres, but its transformed-disparity pictures can be
turned back into numbers (relief_depth.jet_to_scalar), and those numbers are a depth
shape: flat on the road, falling inside the pothole. That is enough to ask a
second-camera question without any new measurement:

    within one pothole and the road around it, does the predicted relief rank the
    pixels the way the stereo shape does?

Per pothole (connected component of the label, >= 150 px), both maps are measured from
a plane fitted on the ring of road around it (the label protocol), then compared by
Spearman rank correlation:

    rho_region   over the pothole and its ring  — includes "the pothole is below the road"
    rho_inside   over the pothole only          — the shape of the bowl itself, the hard part
    below_road   whether the prediction puts the pothole below the road at all

The untouched Depth-Anything-V2 checkpoint is scored the same way, as the "before
fine-tuning" row. Its output is inverse depth, so its sign is flipped.

Caveats: transformed disparity is not relief. Its scale is unknown per image and varies
with distance, so nothing here is in millimetres and potholes cannot be compared with each
other. Rank correlation inside one pothole is the strongest statement the data supports.

Usage:
    python scripts/eval_relief_pothole600.py                      # served networks
    python scripts/eval_relief_pothole600.py --weights a.pth,b.pth --tag v4
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from metric_features import road_plane_deviation                                # noqa: E402

DATA = os.path.join(PROJECT_DIR, "pothole600")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothole600")
MIN_PX = 150
MAX_SAMPLE = 5000
SEED = 20261002


def load_split(split):
    """[(name, bgr, pothole mask, depth shape with deeper = larger, colour distance)]"""
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, split, "rgb", "*.png"))):
        name = os.path.basename(p)
        col = cv2.imread(os.path.join(DATA, split, "tdisp", name))
        lab = cv2.imread(os.path.join(DATA, split, "label", name), cv2.IMREAD_GRAYSCALE)
        bgr = cv2.imread(p)
        if col is None or lab is None or bgr is None:
            continue
        index, dist = R.jet_to_scalar(col)
        out.append((name, bgr, (lab > 127).astype(np.uint8), 255.0 - index.astype(np.float32), dist))
    return out


def base_model():
    """Depth-Anything-V2 ViT-S exactly as released: the starting point of the relief network."""
    import torch
    root = os.path.join(PROJECT_DIR, "Depth-Anything-V2")
    if root not in sys.path:
        sys.path.append(root)
    from depth_anything_v2.dpt import DepthAnythingV2
    m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    m.load_state_dict(torch.load(R.BASE_CKPT, map_location="cpu"))
    return m.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def rho(a, b, rng):
    if len(a) > MAX_SAMPLE:
        i = rng.choice(len(a), MAX_SAMPLE, replace=False)
        a, b = a[i], b[i]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def score(pred, shape, mask, rng):
    """One row per pothole in the frame."""
    n, comp = cv2.connectedComponents(mask)
    rows = []
    for c in range(1, n):
        m = (comp == c).astype(np.uint8)
        if int(m.sum()) < MIN_PX:
            continue
        others = (mask > 0) & (m == 0)                      # other potholes are not road
        t = road_plane_deviation(shape, m, ~others)
        p = road_plane_deviation(pred, m, ~others)
        if t is None or p is None:
            continue
        region = (m > 0) | t[1]
        inside = m > 0
        rows.append({"pothole": c, "px": int(m.sum()),
                     "rho_region": rho(p[0][region], t[0][region], rng),
                     "rho_inside": rho(p[0][inside], t[0][inside], rng),
                     "below_road": bool(np.median(p[0][inside]) > 0),
                     "target_below_road": bool(np.median(t[0][inside]) > 0)})
    return rows


def summarise(d, rng, n_boot=2000):
    out = {"n": int(len(d))}
    for c in ("rho_region", "rho_inside"):
        v = d[c].dropna().values
        boot = [np.median(rng.choice(v, len(v))) for _ in range(n_boot)]
        out[c] = {"median": float(np.median(v)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                  "share_above_0.5": float((v > 0.5).mean())}
    out["below_road"] = float(d.below_road.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=",".join(R.installed_weights()),
                    help="relief checkpoints, comma-separated (averaged)")
    ap.add_argument("--tag", default="served")
    ap.add_argument("--splits", default="validation,testing")
    args = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nets = [R.build_model(w).to(device).eval() for w in args.weights.split(",")]
    base = base_model()
    rng = np.random.default_rng(SEED)

    rows, dists = [], []
    for split in args.splits.split(","):
        frames = load_split(split)
        for name, bgr, mask, shape, dist in frames:
            dists.append(dist)
            preds = {"relief": R.predict_relief(bgr, nets),
                     # inverse depth: larger = closer, so deeper = smaller; flip the sign
                     "da_v2_untouched": -base.infer_image(bgr, 518).astype(np.float32)}
            for method, pred in preds.items():
                for r in score(pred, shape, mask, rng):
                    rows.append({"split": split, "image": name, "method": method, **r})
        print(f"{split}: {len(frames)} frames", flush=True)

    d = pd.DataFrame(rows)
    print(f"\ncolour scale check: mean distance to the JET scale {np.mean(dists):.2f} (of 441), worst frame {np.max(dists):.2f}")
    t = d[d.method == "relief"]
    print(f"stereo shape puts the pothole below the road in {t.target_below_road.mean():.1%} of {len(t)} potholes")
    results = {"weights": args.weights.split(","), "colour_distance_mean": float(np.mean(dists)), "splits": {}}
    print(f"\n{'split':11s} {'method':16s} {'n':>4s} {'rho region':>22s} {'rho inside':>22s} {'inside > 0.5':>13s} {'below road':>11s}")
    for split in args.splits.split(","):
        results["splits"][split] = {}
        for method in ("da_v2_untouched", "relief"):
            s = summarise(d[(d.split == split) & (d.method == method)], rng)
            results["splits"][split][method] = s
            fmt = lambda k: f"{s[k]['median']:.3f} [{s[k]['ci95'][0]:.3f}, {s[k]['ci95'][1]:.3f}]"   # noqa: E731
            print(f"{split:11s} {method:16s} {s['n']:4d} {fmt('rho_region'):>22s} {fmt('rho_inside'):>22s} "
                  f"{s['rho_inside']['share_above_0.5']:13.1%} {s['below_road']:11.1%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    d.to_csv(os.path.join(OUT_DIR, f"relief_shape_potholes_{args.tag}.csv"), index=False)
    out = os.path.join(OUT_DIR, f"relief_shape_eval_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
