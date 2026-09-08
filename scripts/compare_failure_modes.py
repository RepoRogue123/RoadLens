"""
Phase 1 BEFORE/AFTER demonstration — the three depth-failure modes.

Shows, on the same pothole, what the ORIGINAL depth-only pipeline concludes versus
what RoadLens' Phase 1 geometry engine concludes. The three failure modes:

  1. WATER REFLECTION  — water mirrors the sky, so the depth model sees a flat
                         surface and the original rule reports "Shallow".
  2. TEXTURE / SHADOW  — high 2D contrast makes the depth model hallucinate a
                         crater, so the original rule over-reports "Deep".
  3. DRY / SHADOWLESS  — no shading cue, so depth flattens and a real crater is
                         under-reported as "Shallow".

BEFORE verdict  = features.extract_depth_features() -> classifier.classify_severity()
                  (this IS the original repo's depth-only rule-based classifier)
AFTER  verdict  = the same, then corrected with Phase 1 signals that do NOT trust
                  the interior depth: boundary curvature, road-relative bowl depth,
                  surface-normal deviation, the 6-cue water ensemble, and the DINOv2
                  texture-illusion override (the exact logic api.py applies).

Each image -> a 2x2 figure: RGB+mask | depth | BEFORE verdict | AFTER verdict.

Usage:
    python scripts/compare_failure_modes.py                       # default candidate set
    python scripts/compare_failure_modes.py --image <path>.jpg     # a single image
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
from classifier import classify_severity                    # noqa: E402
from features import (                                       # noqa: E402
    extract_depth_features,
    extract_curvature_features,
    extract_depth_profile_features,
    extract_surface_normal_features,
)
from water_detection import detect_water                     # noqa: E402
try:
    from foundation_features import extract_foundation_features
    _HAS_DINO = True
except Exception:
    _HAS_DINO = False

DARK = "#0f172a"
PANEL = "#1e293b"
TEXT = "#e2e8f0"
SEV_COLOR = {"Deep": "#f87171", "Moderate": "#fbbf24", "Shallow": "#34d399",
             "No pothole": "#94a3b8"}

# Default candidates: pic-1/pic-9 are the confirmed water cases; the kaggle picks
# are probed for texture-illusion / dry-flat behavior.
DEFAULT_CANDIDATES = [
    "data1/train/images/pic-1-_jpg.rf.49882cdb272111f43a6656b1494a4918.jpg",
    "data1/train/images/pic-9-_jpg.rf.10d9db0c8fac4eb5b01de9fc71dd19da.jpg",
    "merged_dataset/valid/images/kaggle_pic-114-_jpg.rf.a0f30e06b3b96d7879d5f55a7012433c.jpg",
    "merged_dataset/valid/images/kaggle_pic-123-_jpg.rf.385ae3fbcdabda81f72ddf11f6a4b93d.jpg",
    "merged_dataset/valid/images/kaggle_pic-129-_jpg.rf.d307956eee8ac32fbe793335e87c7b67.jpg",
]


def severity_from_geometry(curv, normals, bowl):
    """Depth-independent severity from boundary shape + road-relative bowl."""
    score = 0
    if curv:
        if curv.get("high_curvature_fraction", 0) > 0.18:
            score += 1
        if curv.get("max_curvature", 0) > 0.30:
            score += 1
    if normals and normals.get("max_normal_deviation", 0) > 25:
        score += 1
    if bowl and bowl.get("max_bowl_depth", 0) > 0.10:
        score += 1
    return "Deep" if score >= 3 else ("Moderate" if score >= 1 else "Shallow")


def decide_after(before, depth_feats, curv, normals, bowl, water, dino):
    """Apply Phase 1 corrections; return (after_verdict, failure_mode, reason)."""
    inside_v = dino.get("dinov2_inside_variance", 0.0) if dino else 0.0
    outside_v = dino.get("dinov2_outside_variance", 0.0) if dino else 0.0
    geo_sev = severity_from_geometry(curv, normals, bowl)
    ldc = depth_feats.get("local_depth_contrast", 0.0)

    # 1. WATER: depth interior is invalid — fall back to boundary geometry
    if water and water.get("is_water"):
        after = geo_sev if geo_sev != "Shallow" else "Moderate"
        return after, "WATER REFLECTION", (
            f"water p={water['water_probability']:.2f} ({water['confidence_level']}). "
            f"Interior depth ignored; severity from boundary geometry.")

    # 2. TEXTURE / SHADOW illusion: 'Deep' but interior smoother than road
    if before == "Deep" and 0 < inside_v < outside_v * 0.9:
        return "Shallow", "TEXTURE / SHADOW ILLUSION", (
            f"DINOv2 inside var {inside_v:.2f} < outside {outside_v:.2f}×0.9 -> "
            f"flat patch faking a crater. Downgraded.")

    # 3. DRY / SHADOWLESS: depth flattened (low contrast) but boundary says crater
    if before == "Shallow" and ldc < 0.06 and geo_sev != "Shallow":
        return geo_sev, "DRY / SHADOWLESS FLATTENING", (
            f"depth contrast {ldc:.3f}≈0 but boundary curvature/normals indicate "
            f"a real crater. Upgraded.")

    return before, "AGREEMENT (no failure)", "depth and geometry agree."


def render(image_path, out_dir):
    masks = get_all_masks(image_path)
    if not masks:
        print(f"  [skip] no pothole detected: {os.path.basename(image_path)}")
        return None
    mask = masks[0]
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    depth = get_depth_map(image_bgr)

    # ---- BEFORE: original depth-only rule ----
    depth_feats = extract_depth_features(mask, depth)
    before = classify_severity(depth_feats)

    # ---- AFTER: Phase 1 geometry + water + illusion ----
    curv = extract_curvature_features(mask)
    bowl = extract_depth_profile_features(mask, depth)
    normals = extract_surface_normal_features(depth, mask)
    water = detect_water(image_rgb, mask, depth_map=depth, curvature_features=curv)
    dino = extract_foundation_features(image_rgb, mask) if _HAS_DINO else None
    after, mode, reason = decide_after(before, depth_feats, curv, normals, bowl, water, dino)

    # ---- figure ----
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=DARK)

    a1.imshow(image_rgb)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        c = c.squeeze()
        if c.ndim == 2:
            a1.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]),
                    color="#fde047", linewidth=2)
    a1.set_title("Input + pothole mask", color=TEXT)
    a1.axis("off")

    im2 = a2.imshow(depth, cmap="inferno")
    a2.set_title("Depth-Anything-V2 depth (the fragile signal)", color=TEXT, fontsize=11)
    a2.axis("off")
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)

    # BEFORE panel
    a3.set_facecolor(PANEL)
    a3.axis("off")
    a3.text(0.5, 0.93, "BEFORE  —  depth-only (original)", transform=a3.transAxes,
            ha="center", color=TEXT, fontsize=13, fontweight="bold")
    bbox_b = (f"max_depth        = {depth_feats['max_depth']:.3f}\n"
              f"local_contrast   = {depth_feats['local_depth_contrast']:.3f}\n"
              f"depth_std        = {depth_feats['depth_std']:.3f}\n"
              f"area             = {depth_feats['area']}")
    a3.text(0.06, 0.62, bbox_b, transform=a3.transAxes, family="monospace",
            color=TEXT, fontsize=11, va="top")
    a3.text(0.5, 0.30, before, transform=a3.transAxes, ha="center",
            color=SEV_COLOR.get(before, TEXT), fontsize=34, fontweight="bold")
    a3.text(0.5, 0.13, "rule on interior depth statistics", transform=a3.transAxes,
            ha="center", color="#94a3b8", fontsize=9, style="italic")

    # AFTER panel
    a4.set_facecolor(PANEL)
    a4.axis("off")
    a4.text(0.5, 0.93, "AFTER  —  RoadLens Phase 1", transform=a4.transAxes,
            ha="center", color=TEXT, fontsize=13, fontweight="bold")
    lines = []
    if curv:
        lines.append(f"high_curv_frac   = {curv['high_curvature_fraction']:.2f}")
        lines.append(f"max_curvature    = {curv['max_curvature']:.3f}")
    if normals:
        lines.append(f"max_normal_dev   = {normals['max_normal_deviation']:.1f} deg")
    if bowl:
        lines.append(f"max_bowl_depth   = {bowl['max_bowl_depth']:.3f}")
    if water:
        lines.append(f"water_prob       = {water['water_probability']:.2f}")
    if dino:
        lines.append(f"dino in/out var  = {dino['dinov2_inside_variance']:.2f}/"
                     f"{dino['dinov2_outside_variance']:.2f}")
    a4.text(0.06, 0.70, "\n".join(lines), transform=a4.transAxes, family="monospace",
            color=TEXT, fontsize=10, va="top")
    a4.text(0.5, 0.34, after, transform=a4.transAxes, ha="center",
            color=SEV_COLOR.get(after, TEXT), fontsize=34, fontweight="bold")
    changed = "corrected ✓" if after != before else "confirmed"
    a4.text(0.5, 0.20, f"depth-independent geometry — {changed}", transform=a4.transAxes,
            ha="center", color="#94a3b8", fontsize=9, style="italic")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(f"FAILURE MODE: {mode}\n{reason}", color=TEXT, fontsize=13, y=0.99)
    out_path = os.path.join(out_dir, f"failure_{stem}.png")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{mode}] {before} -> {after}   saved: {os.path.relpath(out_path, PROJECT_DIR)}")
    return mode, before, after, out_path


def scan(image_dir, limit, only_flips):
    """Probe many images and print a table of BEFORE/AFTER + failure mode, to hunt
    for the most compelling examples. No figures are written."""
    paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if limit:
        paths = paths[:limit]
    print(f"scanning {len(paths)} images in {os.path.relpath(image_dir, PROJECT_DIR)}\n")
    print(f"{'image':<48} {'before':<9} {'after':<9} mode")
    print("-" * 100)
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
        curv = extract_curvature_features(mask)
        bowl = extract_depth_profile_features(mask, depth)
        normals = extract_surface_normal_features(depth, mask)
        water = detect_water(rgb, mask, depth_map=depth, curvature_features=curv)
        dino = extract_foundation_features(rgb, mask) if _HAS_DINO else None
        after, mode, _ = decide_after(before, df, curv, normals, bowl, water, dino)
        if only_flips and after == before:
            continue
        flag = "  <== FLIP" if after != before else ""
        print(f"{os.path.basename(p)[:46]:<48} {before:<9} {after:<9} {mode}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="single image (overrides default set)")
    ap.add_argument("--out-dir", default=os.path.join("ml_results", "failure_modes_demo"))
    ap.add_argument("--scan", default=None, help="directory of .jpg to probe (prints table, no figures)")
    ap.add_argument("--limit", type=int, default=40, help="max images to scan")
    ap.add_argument("--only-flips", action="store_true", help="scan: only print verdict changes")
    args = ap.parse_args()

    if args.scan:
        scan_dir = args.scan if os.path.isabs(args.scan) else os.path.join(PROJECT_DIR, args.scan)
        scan(scan_dir, args.limit, args.only_flips)
        return

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.image:
        cands = [args.image if os.path.isabs(args.image) else os.path.join(PROJECT_DIR, args.image)]
    else:
        cands = [os.path.join(PROJECT_DIR, p) for p in DEFAULT_CANDIDATES]
        cands = [c for c in cands if os.path.isfile(c)]

    print(f"Comparing BEFORE (depth-only) vs AFTER (Phase 1) on {len(cands)} image(s):")
    for c in cands:
        print(f"- {os.path.basename(c)}")
        render(c, out_dir)


if __name__ == "__main__":
    main()
