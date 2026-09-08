"""
Phase 3 diagnostic probe — where does the (partially built) DINOv2
texture-illusion override still behave unreliably?

The override in api.py fires when consensus == "Deep" AND
    dinov2_inside_variance < dinov2_outside_variance * 0.9
This probe measures, per pothole, the quantities that rule depends on, and
flags the cases where the rule is fragile:

  BORDERLINE   ratio within [0.80, 1.05]  -> verdict decided by a hair
  SPARSE-PATCH mask covers < 8 DINOv2 patches -> variance is statistically thin
  ONE-WAY      rule can only downgrade "Deep"; never rescues an under-report

Usage:
    python scripts/probe_semantic_gaps.py --dir data1/train/images --limit 25
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks                    # noqa: E402
from inference import get_depth_map                       # noqa: E402
from classifier import classify_severity                  # noqa: E402
from features import extract_depth_features               # noqa: E402
from foundation_features import extract_foundation_features  # noqa: E402

PATCH = 14  # DINOv2 patch grid is 14x14 over the resized input

OVERRIDE_RATIO = 0.9
BORDERLINE_LO, BORDERLINE_HI = 0.80, 1.05
MIN_PATCHES = 8


def patches_covered(mask):
    """How many of the 14x14 DINOv2 patches the mask actually covers."""
    small = cv2.resize(mask.astype(np.uint8), (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    return int((small > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data1/train/images")
    ap.add_argument("--limit", type=int, default=25)
    args = ap.parse_args()

    d = args.dir if os.path.isabs(args.dir) else os.path.join(PROJECT_DIR, args.dir)
    paths = sorted(glob.glob(os.path.join(d, "*.jpg")))[: args.limit]

    print(f"probing {len(paths)} images\n")
    print(f"{'image':<40} {'before':<9} {'in_var':>7} {'out_var':>7} {'raw':>6} "
          f"{'ring':>7} {'corr':>6} {'patch':>5}  flags")
    print("-" * 118)
    raw_ratios, corr_ratios = [], []

    for p in paths:
        masks = get_all_masks(p)
        if not masks:
            continue
        mask = masks[0]
        bgr = cv2.imread(p)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = get_depth_map(bgr)

        df = extract_depth_features(mask, depth)
        if df is None:
            continue
        before = classify_severity(df)

        feats = extract_foundation_features(rgb, mask)
        if not feats:
            continue
        iv = feats["dinov2_inside_variance"]
        ov = feats["dinov2_outside_variance"]
        ratio = iv / ov if ov > 1e-9 else float("nan")
        npatch = patches_covered(mask)

        flags = []
        if BORDERLINE_LO <= ratio <= BORDERLINE_HI:
            flags.append("BORDERLINE")
        if npatch < MIN_PATCHES:
            flags.append("SPARSE-PATCH")
        fires = before == "Deep" and iv < ov * OVERRIDE_RATIO
        if fires:
            flags.append("OVERRIDE-FIRES")
        if before != "Deep" and ratio < OVERRIDE_RATIO:
            # smooth interior but the rule cannot act — one-way blindness
            flags.append("ONE-WAY-MISS")

        ring = feats.get("dinov2_ring_variance", float("nan"))
        corr = feats.get("dinov2_ratio_corrected", float("nan"))
        raw_ratios.append(ratio)
        if corr == corr:  # not NaN
            corr_ratios.append(corr)

        print(f"{os.path.basename(p)[:38]:<40} {before:<9} {iv:>7.3f} {ov:>7.3f} "
              f"{ratio:>6.2f} {ring:>7.3f} {corr:>6.2f} {npatch:>5}  {','.join(flags) or '-'}")

    _summarise(raw_ratios, corr_ratios)


def _summarise(raw, corr):
    """Compare the raw (whole-scene) ratio against the corrected (ring, size-matched)."""
    import statistics as st
    print()
    for name, vals in (("raw  (whole scene)", raw), ("corr (ring, matched)", corr)):
        vals = [v for v in vals if v == v]
        if not vals:
            continue
        below = sum(1 for v in vals if v < OVERRIDE_RATIO)
        print(f"{name}:  n={len(vals):<3} range {min(vals):.2f}–{max(vals):.2f}  "
              f"median {st.median(vals):.2f}  spread {max(vals)-min(vals):.2f}  "
              f"below {OVERRIDE_RATIO}: {below}/{len(vals)}")
    print("\nA useful correction widens the spread and stops nearly everything "
          "falling on one side of the cut-off.")


if __name__ == "__main__":
    main()
