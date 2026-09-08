"""
Phase 3 diagnostic figures — the open problems in the semantic override.

Produces two figures documenting why the DINOv2 texture-illusion override is
implemented but not yet calibrated:

  1. phase3_threshold_instability.png
     The SAME pothole under three augmentations lands on both sides of the
     hard 0.9 variance-ratio threshold (0.84 / 0.92 / 0.95). The verdict is
     decided by augmentation noise, not by the scene.

  2. phase3_ratio_distribution.png
     The measured inside/outside variance ratio across the probe set sits far
     below the 0.9 cut-off for nearly every sample, so the rule as written
     would downgrade almost any "Deep" verdict. The threshold needs to be
     learned from data, not hand-set.

Usage:  python scripts/phase3_diagnostics.py
"""
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
from foundation_features import extract_foundation_features   # noqa: E402

DARK = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
THRESHOLD = 0.9
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "phase3_diagnostics")

# Three augmentations of the same physical pothole
TRIPLET_GLOB = "data1/train/images/pic-100-*.jpg"
# Wider set for the distribution figure
DIST_GLOB = "data1/train/images/pic-1*.jpg"
DIST_LIMIT = 18

PATCH = 14


def analyze(path):
    masks = get_all_masks(path)
    if not masks:
        return None
    mask = masks[0]
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    feats = extract_foundation_features(rgb, mask)
    if not feats:
        return None
    iv = feats["dinov2_inside_variance"]
    ov = feats["dinov2_outside_variance"]
    small = cv2.resize(mask.astype(np.uint8), (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    return {
        "rgb": rgb, "mask": mask, "inside": iv, "outside": ov,
        "ratio": iv / ov if ov > 1e-9 else float("nan"),
        "patches": int((small > 0).sum()),
        "name": os.path.basename(path),
    }


def fig_threshold_instability(records):
    fig, axes = plt.subplots(2, len(records), figsize=(4.6 * len(records), 8.4), facecolor=DARK)
    if len(records) == 1:
        axes = axes.reshape(2, 1)

    for i, r in enumerate(records):
        ax = axes[0, i]
        ax.imshow(r["rgb"])
        cnts, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            c = c.squeeze()
            if c.ndim == 2:
                ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), color="#fde047", lw=2)
        ax.set_title(f"Augmentation {i + 1}", color=TEXT, fontsize=12)
        ax.axis("off")

        ax = axes[1, i]
        ax.set_facecolor(PANEL)
        fires = r["ratio"] < THRESHOLD
        col = "#f87171" if fires else "#34d399"
        ax.barh([1, 0], [r["outside"], r["inside"]], color=["#64748b", col], height=0.55)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["inside", "road"], color=TEXT, fontsize=10)
        ax.set_xlim(0, max(r["outside"], r["inside"]) * 1.25)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#334155")
        ax.set_xlabel("DINOv2 feature variance", color="#94a3b8", fontsize=10)
        verdict = "OVERRIDE FIRES\n→ downgraded to Shallow" if fires else "NO OVERRIDE\n→ verdict unchanged"
        ax.set_title(
            f"ratio = {r['ratio']:.2f}   (threshold {THRESHOLD})\n{verdict}",
            color=col, fontsize=11, fontweight="bold", pad=10,
        )

    fig.suptitle(
        "PHASE 3 · OPEN PROBLEM 1 — the 0.9 threshold is not calibrated\n"
        "One physical pothole, three augmentations: the variance ratio straddles the cut-off, "
        "so the same defect receives opposite verdicts.",
        color=TEXT, fontsize=13, y=0.98,
    )
    out = os.path.join(OUT_DIR, "phase3_threshold_instability.png")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(out, PROJECT_DIR))


def fig_ratio_distribution(ratios, labels):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK)
    ax.set_facecolor(PANEL)
    order = np.argsort(ratios)
    r = np.array(ratios)[order]
    cols = ["#f87171" if v < THRESHOLD else "#34d399" for v in r]
    ax.bar(range(len(r)), r, color=cols)
    ax.axhline(THRESHOLD, color="#fbbf24", ls="--", lw=2)
    ax.text(len(r) * 0.5, THRESHOLD + 0.03, f"hand-set threshold = {THRESHOLD}",
            color="#fbbf24", fontsize=11, ha="center")
    ax.set_ylabel("inside / outside variance ratio", color=TEXT, fontsize=11)
    ax.set_xlabel(f"samples (n={len(r)}), sorted", color="#94a3b8", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#94a3b8")
    for s in ax.spines.values():
        s.set_color("#334155")
    below = int((r < THRESHOLD).sum())
    ax.set_title(
        f"PHASE 3 · OPEN PROBLEM 2 — the rule would fire on almost everything\n"
        f"{below}/{len(r)} samples fall below the cut-off; the observed ratio range is "
        f"{r.min():.2f}–{r.max():.2f}, so 0.9 does not separate illusions from real craters.",
        color=TEXT, fontsize=13, pad=14,
    )
    out = os.path.join(OUT_DIR, "phase3_ratio_distribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("saved", os.path.relpath(out, PROJECT_DIR))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    trip = sorted(glob.glob(os.path.join(PROJECT_DIR, TRIPLET_GLOB)))[:3]
    recs = [x for x in (analyze(p) for p in trip) if x]
    if recs:
        for r in recs:
            print(f"  {r['name'][:40]:<42} ratio={r['ratio']:.2f} patches={r['patches']}")
        fig_threshold_instability(recs)

    paths = sorted(glob.glob(os.path.join(PROJECT_DIR, DIST_GLOB)))[:DIST_LIMIT]
    ratios, labels = [], []
    for p in paths:
        rec = analyze(p)
        if rec:
            ratios.append(rec["ratio"])
            labels.append(rec["name"])
    if ratios:
        fig_ratio_distribution(ratios, labels)


if __name__ == "__main__":
    main()
