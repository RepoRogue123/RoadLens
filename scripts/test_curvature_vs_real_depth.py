"""
M2.2 — Does boundary curvature predict REAL depth, once the window bug is fixed?

Background
----------
Phase 1's headline claim is that boundary curvature encodes severity: deep
potholes have steep walls, which produce sharp high-curvature outlines.

Measured against RealSense ground truth, mean_curvature correlated -0.254 with
true depth — the wrong sign. But controlling for pothole area collapsed it to
-0.041, which pointed at an artefact rather than physics: the Savitzky-Golay
window was a CONSTANT `min(n-1, 31)`, so a small contour was smoothed almost
away while a large one was barely touched. Larger potholes are deeper, so
curvature fell as depth rose for purely procedural reasons.

features._curvature_window now scales the window with contour length. This script
settles whether that was the whole story.

It runs both window modes over the same potholes and reports, for each curvature
feature: raw correlation with true depth, and partial correlation controlling for
log(area). Curvature needs no depth model at all, so this is fast.

Read the PARTIAL column. The raw column is contaminated by size in both modes.

Usage:
    python scripts/test_curvature_vs_real_depth.py
"""
import csv
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import features as F                                       # noqa: E402
from scripts.pothrgbd_metric_labels import (               # noqa: E402
    DATA_DIR, load_polygons, timestamp_key, road_plane_deviation,
    DEPTH_VALID_MIN, MIN_MASK_PX, MAX_PLAUSIBLE_MM, MIN_PLAUSIBLE_MM,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
CURV_COLS = [
    "max_curvature", "mean_curvature", "std_curvature", "p90_curvature",
    "high_curvature_fraction", "curvature_entropy", "concave_fraction",
    "curvature_sign_changes", "contour_length", "contour_elongation",
]


def partial_corr(x, y, z):
    """Correlation of x and y after linearly removing z from both."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if x.size < 5 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan"), float("nan")
    raw = float(np.corrcoef(x, y)[0, 1])
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return raw, float("nan")
    return raw, float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    imgs = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.jpg")))
    deps = {timestamp_key(p): p
            for p in glob.glob(os.path.join(DATA_DIR, "depths", "*.npy"))}
    labs = {timestamp_key(p): p
            for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}

    print(f"Curvature vs REAL measured depth — {len(imgs)} frames\n")

    rows = []
    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        if not dp or not lp:
            continue
        gt_depth = np.load(dp)
        h, w = gt_depth.shape[:2]
        for j, m in enumerate(load_polygons(lp, h, w), start=1):
            res = road_plane_deviation(gt_depth, m, valid_min=DEPTH_VALID_MIN)
            if res is None:
                continue
            dev, _n, rms = res
            if rms > 40.0:
                continue
            inside = dev[(m > 0) & (gt_depth >= DEPTH_VALID_MIN)]
            if inside.size < MIN_MASK_PX // 2:
                continue
            gt = float(np.percentile(inside, 90))
            if not (MIN_PLAUSIBLE_MM <= gt <= MAX_PLAUSIBLE_MM):
                continue

            rec = {"key": k, "pothole": j, "gt_mm": gt, "area_px": int(m.sum())}
            for mode in ("constant", "fractional"):
                F.CURVATURE_WINDOW_MODE = mode
                cf = F.extract_curvature_features(m)
                if cf:
                    for c in CURV_COLS:
                        rec[f"{mode}_{c}"] = float(cf.get(c, np.nan))
            rows.append(rec)

        if i % 200 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames, {len(rows)} potholes", flush=True)

    F.CURVATURE_WINDOW_MODE = "fractional"      # restore

    if not rows:
        print("nothing measured")
        return

    gt = np.array([r["gt_mm"] for r in rows])
    la = np.log(np.array([r["area_px"] for r in rows], dtype=float))

    print(f"\n{len(rows)} potholes, {len({r['key'] for r in rows})} frames")
    print(f"  true depth median {np.median(gt):.1f}mm\n")

    r_area = float(np.corrcoef(la, gt)[0, 1])
    print(f"  REFERENCE  log(area) vs true depth   r = {r_area:+.3f}")
    print("  (any curvature feature that cannot beat this is not earning its place)\n")

    hdr = f"  {'feature':<26}{'CONSTANT window':>22}{'FRACTIONAL window':>24}"
    print(hdr)
    print(f"  {'':<26}{'raw':>11}{'partial':>11}{'raw':>12}{'partial':>12}")
    print("  " + "-" * (len(hdr) - 2))

    best = 0.0
    for c in CURV_COLS:
        out = [c]
        for mode in ("constant", "fractional"):
            v = np.array([r.get(f"{mode}_{c}", np.nan) for r in rows], float)
            raw, par = partial_corr(v, gt, la)
            out += [raw, par]
            if mode == "fractional" and np.isfinite(par):
                best = max(best, abs(par))
        print(f"  {out[0]:<26}{out[1]:>11.3f}{out[2]:>11.3f}"
              f"{out[3]:>12.3f}{out[4]:>12.3f}")

    print(f"\n  Strongest |partial r| under the fixed window: {best:.3f}")
    if best < abs(r_area):
        print(f"  This is BELOW log(area)'s {abs(r_area):.3f}. Fixing the window did not")
        print("  reveal a depth signal in curvature. Phase 1's claim that boundary")
        print("  curvature encodes DEPTH is not supported by measured data.")
        print("  Curvature never looks inside the pothole, so this is the expected")
        print("  physical result rather than a bug — the claim, not the code, is wrong.")
    else:
        print("  Curvature beats log(area) after the fix — the original negative")
        print("  result WAS the window artefact. Phase 1's claim survives.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "curvature_window_test.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n  wrote {os.path.relpath(out_csv, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
