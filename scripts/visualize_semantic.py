"""
Phase 3 isolation visualizer — Semantic Intelligence (DINOv2) + Texture-Illusion
Override.

Renders a 3-panel figure:

  1. DINOv2 inside-vs-outside feature variance + semantic dissimilarity bars
     (reuses foundation_features.extract_foundation_features). Real craters have
     HIGHER internal variance than the road; a flat textured patch does not.
  2. Raw Depth-Anything-V2 depth (what a 2D texture illusion can fake as "Deep").
  3. Illusion-corrected depth: inside the mask flattened to
     depth*0.15 + road_mean*0.85 — the EXACT operation api.py applies when
     inside_variance < outside_variance * 0.9 on a "Deep" prediction.

Needs YOLO + Depth-Anything-V2 + DINOv2 (facebook/dinov2-base, downloaded via
transformers on first run). If DINOv2 can't load, the script exits with a clear
message and Phase 3 falls back to documenting the command only.

Usage:
    python scripts/visualize_semantic.py
    python scripts/visualize_semantic.py --image <path>
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

from segmentation import get_all_masks                       # noqa: E402
from inference import get_depth_map                          # noqa: E402
from foundation_features import extract_foundation_features  # noqa: E402

DARK = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"


def pick_default_image():
    for pat in ("merged_dataset/valid/images/kaggle_pic-114*.jpg",
                "merged_dataset/valid/images/kaggle_pic-*.jpg",
                "merged_dataset/valid/images/*.jpg"):
        hits = sorted(glob.glob(os.path.join(PROJECT_DIR, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError("No sample image found.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--out-dir", default=os.path.join("ml_results", "semantic_demo"))
    args = ap.parse_args()

    image_path = args.image or pick_default_image()
    image_path = image_path if os.path.isabs(image_path) else os.path.join(PROJECT_DIR, image_path)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[semantic] image: {image_path}")
    masks = get_all_masks(image_path)
    if not masks:
        print("[semantic] No potholes detected. Try another --image.")
        sys.exit(1)
    mask = masks[0]
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("[semantic] loading DINOv2 (facebook/dinov2-base, first run downloads)...")
    feats = extract_foundation_features(image_rgb, mask)
    if feats is None:
        print("[semantic] DINOv2 unavailable (no model/cache or load failed). "
              "Phase 3 image cannot be generated on this machine.")
        sys.exit(2)

    inside_var = feats.get("dinov2_inside_variance", 0.0)
    outside_var = feats.get("dinov2_outside_variance", 0.0)
    dissim = feats.get("dinov2_dissimilarity", 0.0)
    illusion = 0 < inside_var < (outside_var * 0.9)

    print("[semantic] estimating depth (Depth-Anything-V2)...")
    depth = get_depth_map(image_bgr)
    # Illusion-corrected depth (exact api.py operation)
    road_mean = np.mean(depth[mask == 0]) if np.any(mask == 0) else np.mean(depth)
    corrected = np.where(mask > 0, depth * 0.15 + road_mean * 0.85, depth)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.6), facecolor=DARK)

    # 1. DINOv2 variance / dissimilarity bars
    a1.set_facecolor(PANEL)
    labels = ["inside\nvariance", "outside\nvariance", "dissimilarity"]
    vals = [inside_var, outside_var, dissim]
    colors = ["#f87171", "#38bdf8", "#a78bfa"]
    bars = a1.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}",
                ha="center", va="bottom", color=TEXT, fontsize=9)
    verdict = ("TEXTURE ILLUSION\n(inside < outside × 0.9 → downgrade to Shallow)"
               if illusion else
               "Genuine crater signature\n(internal variance ≥ road)")
    a1.set_title("DINOv2 semantic features\n" + verdict, color=TEXT, fontsize=10)
    a1.tick_params(colors=TEXT)
    for sp in a1.spines.values():
        sp.set_color("#334155")

    # 2. raw depth
    im2 = a2.imshow(depth, cmap="inferno")
    a2.set_title("Raw depth (can be faked by 2D texture)", color=TEXT, fontsize=11)
    a2.axis("off")
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)

    # 3. corrected depth
    im3 = a3.imshow(corrected, cmap="inferno", vmin=float(depth.min()), vmax=float(depth.max()))
    tag = "APPLIED" if illusion else "(no override triggered here)"
    a3.set_title(f"Illusion-corrected depth {tag}", color=TEXT, fontsize=11)
    a3.axis("off")
    fig.colorbar(im3, ax=a3, fraction=0.046, pad=0.02)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle("RoadLens — DINOv2 Texture-Illusion Override", color=TEXT, fontsize=14)
    out_path = os.path.join(out_dir, f"semantic_{stem}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"[semantic] saved: {out_path}  (illusion_triggered={illusion})")


if __name__ == "__main__":
    main()
