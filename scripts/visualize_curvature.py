"""
Phase 1 (hero) isolation visualizer — Curvature-Invariant Geometry.

Renders the boundary-curvature analysis that powers RoadLens' depth-independent
severity features. Reuses the SAME contour smoothing + discrete signed curvature
math as features.extract_curvature_features(), then draws:

  Left  : pothole contour overlaid on the image, each boundary point colored by
          |curvature| (steep walls = bright = high curvature).
  Right : signed curvature vs normalized arc-length, concave regions shaded,
          annotated with the 10 curvature features.

Needs ONLY the YOLO segmentation model (no depth model) — it is computed purely
from the 2D mask boundary, which is the whole point of the design.

Usage:
    python scripts/visualize_curvature.py
    python scripts/visualize_curvature.py --image merged_dataset/valid/images/kaggle_pic-114-_jpg.rf.a0f30e06b3b96d7879d5f55a7012433c.jpg
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Make repo root importable when run from anywhere
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks            # noqa: E402
from features import extract_curvature_features    # noqa: E402

DARK = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"


def signed_curvature_profile(mask):
    """Replicate the contour smoothing + signed curvature from features.py so we
    can plot the per-point profile (the library function only returns summaries)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    pts = contour.squeeze()
    if pts.ndim != 2 or len(pts) < 15:
        return None

    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)

    win = min(len(x) - 1, 31)
    if win % 2 == 0:
        win -= 1
    win = max(win, 5)

    xs = scipy.signal.savgol_filter(x, window_length=win, polyorder=3, mode="wrap")
    ys = scipy.signal.savgol_filter(y, window_length=win, polyorder=3, mode="wrap")

    dx, dy = np.gradient(xs), np.gradient(ys)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    denom[denom < 1e-10] = 1e-10
    kappa = (dx * ddy - dy * ddx) / denom
    return xs, ys, kappa


def pick_default_image():
    for pat in (
        "merged_dataset/valid/images/kaggle_pic-114*.jpg",
        "merged_dataset/valid/images/kaggle_pic-*.jpg",
        "merged_dataset/valid/images/*.jpg",
    ):
        hits = sorted(glob.glob(os.path.join(PROJECT_DIR, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError("No sample image found under merged_dataset/valid/images/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="input image path")
    ap.add_argument("--out-dir", default=os.path.join("ml_results", "curvature_demo"))
    args = ap.parse_args()

    image_path = args.image or pick_default_image()
    image_path = image_path if os.path.isabs(image_path) else os.path.join(PROJECT_DIR, image_path)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[curvature] image: {image_path}")
    masks = get_all_masks(image_path)
    if not masks:
        print("[curvature] No potholes detected in this image. Try another --image.")
        sys.exit(1)

    mask = masks[0]  # largest
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    prof = signed_curvature_profile(mask)
    feats = extract_curvature_features(mask)
    if prof is None or feats is None:
        print("[curvature] Contour too small for curvature analysis. Try another --image.")
        sys.exit(1)
    xs, ys, kappa = prof
    abs_k = np.abs(kappa)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=DARK)

    # ---- Left: contour colored by |curvature| ----
    axL.imshow(image_rgb)
    seg_pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([seg_pts[:-1], seg_pts[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=3.0)
    lc.set_array(abs_k[:-1])
    axL.add_collection(lc)
    # mark the highest-curvature points (steepest walls)
    thr = feats["mean_curvature"] + feats["std_curvature"]
    hi = abs_k > thr
    axL.scatter(xs[hi], ys[hi], s=18, c="#fde047", edgecolors="black",
                linewidths=0.4, zorder=5, label="high-curvature (steep wall)")
    axL.set_title("Boundary curvature (depth-independent)", color=TEXT, fontsize=13)
    axL.axis("off")
    axL.legend(loc="lower right", fontsize=8, framealpha=0.6)
    cb = fig.colorbar(lc, ax=axL, fraction=0.046, pad=0.02)
    cb.set_label("|curvature|", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=TEXT)
    plt.setp(cb.ax.get_yticklabels(), color=TEXT)

    # ---- Right: signed curvature vs arc-length ----
    axR.set_facecolor(PANEL)
    s = np.linspace(0, 1, len(kappa))
    axR.plot(s, kappa, color="#38bdf8", linewidth=1.6, label="signed κ")
    axR.fill_between(s, kappa, 0, where=kappa < 0, color="#f87171", alpha=0.45,
                     label="concave (κ<0)")
    axR.fill_between(s, kappa, 0, where=kappa >= 0, color="#34d399", alpha=0.25,
                     label="convex (κ≥0)")
    axR.axhline(0, color=TEXT, linewidth=0.7, alpha=0.5)
    axR.set_xlabel("normalized arc-length around boundary", color=TEXT)
    axR.set_ylabel("signed curvature κ", color=TEXT)
    axR.set_title("Curvature profile", color=TEXT, fontsize=13)
    axR.tick_params(colors=TEXT)
    for spine in axR.spines.values():
        spine.set_color("#334155")
    axR.legend(loc="upper right", fontsize=8, framealpha=0.6)

    summary = (
        f"max κ      = {feats['max_curvature']:.4f}\n"
        f"mean κ     = {feats['mean_curvature']:.4f}\n"
        f"p90 κ      = {feats['p90_curvature']:.4f}\n"
        f"high-κ frac= {feats['high_curvature_fraction']:.2f}\n"
        f"entropy    = {feats['curvature_entropy']:.2f}\n"
        f"concave    = {feats['concave_fraction']:.2f}\n"
        f"sign chg   = {feats['curvature_sign_changes']}\n"
        f"elongation = {feats['contour_elongation']:.2f}"
    )
    axR.text(0.02, 0.03, summary, transform=axR.transAxes, fontsize=8.5,
             family="monospace", color=TEXT, va="bottom",
             bbox=dict(boxstyle="round", facecolor=DARK, edgecolor="#334155", alpha=0.85))

    stem = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle("RoadLens — Curvature-Invariant Geometry (2D boundary only, no depth)",
                 color=TEXT, fontsize=14, y=0.99)
    out_path = os.path.join(out_dir, f"curvature_{stem}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"[curvature] saved: {out_path}")


if __name__ == "__main__":
    main()
