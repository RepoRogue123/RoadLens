"""
Phase 1 isolation visualizer — Surface Normals + Depth-Profile Bowl Fitting.

Two of the depth-derived (but road-relative) geometry signals in the 39-feature
vector. Renders a 4-panel figure:

  1. Original + pothole mask outline
  2. Depth-Anything-V2 depth heatmap
  3. Surface-normal angular-deviation field (steep walls deviate most from the
     road reference normal) — reuses the math from
     features.extract_surface_normal_features()
  4. Bowl-depth profile: 8 angular cross-sections through the centroid with the
     quadratic road-surface extrapolation, the basis of
     features.extract_depth_profile_features()

Needs the YOLO seg model + Depth-Anything-V2 checkpoint.

Usage:
    python scripts/visualize_geometry.py
    python scripts/visualize_geometry.py --image <path>
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks                          # noqa: E402
from inference import get_depth_map                             # noqa: E402
from features import (                                          # noqa: E402
    extract_surface_normal_features,
    extract_depth_profile_features,
)

DARK = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"


def normal_deviation_field(depth_map, mask):
    """Per-pixel angular deviation (deg) of the surface normal from the road
    reference normal — same construction as extract_surface_normal_features."""
    d = depth_map.astype(np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    d = (d - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(d)

    d_smooth = cv2.GaussianBlur(d, (15, 15), 0)
    dzdx = cv2.Sobel(d_smooth, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(d_smooth, cv2.CV_32F, 0, 1, ksize=3)
    nf = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1.0)
    nx, ny, nz = -dzdx / nf, -dzdy / nf, 1.0 / nf

    road = (mask == 0)
    rnx, rny, rnz = nx[road].mean(), ny[road].mean(), nz[road].mean()
    rn = np.sqrt(rnx ** 2 + rny ** 2 + rnz ** 2) + 1e-8
    rnx, rny, rnz = rnx / rn, rny / rn, rnz / rn

    dot = np.clip(nx * rnx + ny * rny + nz * rnz, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def slice_curves(mask, depth_map, n_slices=8):
    """Reproduce the cross-section sampling so the road-fit curves can be drawn."""
    d = depth_map.astype(np.float32)
    dmin, dmax = float(d.min()), float(d.max())
    d = (d - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(d)
    h, w = mask.shape[:2]
    ys, xs = np.where(mask > 0)
    cy, cx = ys.mean(), xs.mean()
    center_depth = float(d[int(cy), int(cx)])
    margin = max(ys.max() - ys.min(), xs.max() - xs.min())
    curves = []
    for i in range(n_slices):
        a = np.pi * i / n_slices
        ca, sa = np.cos(a), np.sin(a)
        max_dist = int(margin * 1.5)
        t_out, d_out = [], []
        for t in range(-max_dist, max_dist + 1):
            px, py = int(cx + t * ca), int(cy + t * sa)
            if 0 <= px < w and 0 <= py < h and mask[py, px] == 0:
                t_out.append(t)
                d_out.append(float(d[py, px]))
        if len(t_out) < 6:
            continue
        t_arr = np.array(t_out, float)
        d_arr = np.array(d_out, float)
        try:
            coeffs = np.polyfit(t_arr, d_arr, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue
        tt = np.linspace(t_arr.min(), t_arr.max(), 80)
        curves.append((np.degrees(a), t_arr, d_arr, tt, np.polyval(coeffs, tt),
                       float(coeffs[2]), center_depth))
    return curves


def pick_default_image():
    for pat in ("merged_dataset/valid/images/kaggle_pic-123*.jpg",
                "merged_dataset/valid/images/kaggle_pic-*.jpg",
                "merged_dataset/valid/images/*.jpg"):
        hits = sorted(glob.glob(os.path.join(PROJECT_DIR, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError("No sample image found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--out-dir", default=os.path.join("ml_results", "geometry_demo"))
    args = ap.parse_args()

    image_path = args.image or pick_default_image()
    image_path = image_path if os.path.isabs(image_path) else os.path.join(PROJECT_DIR, image_path)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[geometry] image: {image_path}")
    masks = get_all_masks(image_path)
    if not masks:
        print("[geometry] No potholes detected. Try another --image.")
        sys.exit(1)
    mask = masks[0]
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("[geometry] estimating depth (Depth-Anything-V2)...")
    depth = get_depth_map(image_bgr)

    nf = extract_surface_normal_features(depth, mask)
    pf = extract_depth_profile_features(mask, depth)
    dev_field = normal_deviation_field(depth, mask)
    curves = slice_curves(mask, depth)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor=DARK)
    (a1, a2), (a3, a4) = axes

    # 1. original + mask outline
    a1.imshow(image_rgb)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        c = c.squeeze()
        if c.ndim == 2:
            a1.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                    color="#fde047", linewidth=2)
    a1.set_title("Input + pothole mask", color=TEXT)
    a1.axis("off")

    # 2. depth heatmap
    im2 = a2.imshow(depth, cmap="inferno")
    a2.set_title("Depth-Anything-V2 depth", color=TEXT)
    a2.axis("off")
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)

    # 3. normal deviation field (mask the road to emphasize the walls)
    dev_show = np.where(cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2) > 0,
                        dev_field, np.nan)
    im3 = a3.imshow(image_rgb)
    im3 = a3.imshow(dev_show, cmap="turbo", alpha=0.85, vmin=0,
                    vmax=float(np.nanpercentile(dev_show, 98)) if np.isfinite(dev_show).any() else 1)
    title3 = "Surface-normal deviation (deg)"
    if nf:
        title3 += f"   mean={nf['mean_normal_deviation']:.1f}°  max={nf['max_normal_deviation']:.1f}°"
    a3.set_title(title3, color=TEXT, fontsize=11)
    a3.axis("off")
    cb3 = fig.colorbar(im3, ax=a3, fraction=0.046, pad=0.02)
    cb3.set_label("deg from road normal", color=TEXT)
    plt.setp(cb3.ax.get_yticklabels(), color=TEXT)

    # 4. bowl-depth cross-sections
    a4.set_facecolor(PANEL)
    if curves:
        cmap = plt.cm.viridis(np.linspace(0, 1, len(curves)))
        for (ang, t_arr, d_arr, tt, road_fit, road0, cd), col in zip(curves, cmap):
            a4.plot(tt, road_fit, "--", color=col, linewidth=1.3, alpha=0.9)
            a4.scatter(t_arr, d_arr, s=4, color=col, alpha=0.35)
        a4.axvline(0, color="#fde047", linewidth=1.2, label="pothole center")
        a4.scatter([0], [curves[0][6]], color="#f87171", zorder=6, s=40,
                   label="measured center depth")
    a4.set_xlabel("offset from centroid (px)", color=TEXT)
    a4.set_ylabel("normalized depth", color=TEXT)
    title4 = "8-section road-surface extrapolation (bowl fit)"
    if pf:
        title4 += f"\nmean bowl={pf['mean_bowl_depth']:.3f}  max bowl={pf['max_bowl_depth']:.3f}"
    a4.set_title(title4, color=TEXT, fontsize=11)
    a4.tick_params(colors=TEXT)
    for sp in a4.spines.values():
        sp.set_color("#334155")
    a4.legend(loc="upper right", fontsize=8, framealpha=0.6)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle("RoadLens — Surface Normals & Bowl-Depth Profiling (road-relative geometry)",
                 color=TEXT, fontsize=14)
    out_path = os.path.join(out_dir, f"geometry_{stem}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"[geometry] saved: {out_path}")


if __name__ == "__main__":
    main()
