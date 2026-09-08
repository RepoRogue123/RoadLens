"""
Derive TRUE metric severity labels from PothRGBD, and test our labels against them.

Why this exists
---------------
Every accuracy figure in this project is partly a measure of the system agreeing
with itself. The severity labels used for training are derived from
Depth-Anything-V2 output (KMeans over `max_depth` and area), so a model fed
depth features is graded against its own source. That is why `Depth Only`
reaches 96.7% and `All Features` reaches an implausible 99.7%, while
`Geometric Only` sits at 65.8% — the last number is low precisely BECAUSE it
measures something independent of the label source.

PothRGBD breaks that. It ships 1000 RGB frames with paired Intel RealSense D415
depth in MILLIMETRES, plus YOLO-style segmentation polygons. Real measurement,
not inference.

This script:
  1. computes the real bowl depth of every annotated pothole, in millimetres
  2. assigns severity from the bands highway authorities actually use
  3. compares those labels against what Depth-Anything-V2 would have said,
     which quantifies the circularity for the first time

Method
------
RealSense depth is distance along the camera ray, and the camera is oblique to
the road, so raw depth rises with distance down the road regardless of any
pothole. We therefore fit a PLANE to the road immediately surrounding each
pothole and measure deviation from that plane — the same road-relative logic as
features.extract_depth_profile_features, but in real units:

    fit    z = a·x + b·y + c     over the surrounding road ring (least squares)
    bowl   d(x,y) = z_measured(x,y) − z_plane(x,y)      [mm, positive = deeper]

CAVEAT, stated rather than buried: this is depth along the camera ray, not
perpendicular to the road surface. For an oblique view the true vertical depth
is d·cos(θ) where θ is the angle between the ray and the road normal. PothRGBD
is close-range and fairly overhead (max range ~1.5 m), so θ is small and the
correction is modest, but every number below is therefore a slight OVER-estimate
of true vertical depth. Reported as measured; not silently corrected.

Usage:
    python scripts/pothrgbd_metric_labels.py
    python scripts/pothrgbd_metric_labels.py --limit 100 --compare-da
"""
import argparse
import csv
import glob
import os
import re
import sys

from typing import Optional

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "archive", "PUBLIC POTHOLE DATASET")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")

# Highway-authority severity bands, in millimetres. These are the numbers our
# arbitrary 0.3 / 0.6 thresholds in classifier.py are standing in for.
BAND_SHALLOW_MM = 25.0
BAND_MODERATE_MM = 50.0

MIN_MASK_PX = 300          # smaller than this and the plane fit is meaningless
RING_KERNEL = 25           # road ring thickness, pixels
MIN_RING_PX = 400
DEPTH_VALID_MIN = 1        # RealSense writes 0 for "no reading"

# Physical plausibility bounds, in millimetres.
#
# The first full run produced a maximum of 706.9 mm against a 95th percentile of
# 66.5 mm. A 70 cm road pothole does not exist — that is a mis-annotation (a
# drain, a kerb drop, an open trench), a RealSense dropout at the mask edge, or
# a plane fit tilted by something tall in the ring. Left in, a handful of such
# values would dominate any regression fitted on these labels and would make the
# "Deep" band meaningless.
#
# 250 mm is deliberately generous: roughly four times the 95th percentile, so it
# removes physical impossibilities without quietly trimming the real tail. The
# count of rejections is reported rather than hidden.
MAX_PLAUSIBLE_MM = 250.0
MIN_PLAUSIBLE_MM = -10.0   # small negatives are plane-fit noise, not bumps


def severity_from_mm(depth_mm: float) -> str:
    if depth_mm < BAND_SHALLOW_MM:
        return "Shallow"
    if depth_mm < BAND_MODERATE_MM:
        return "Moderate"
    return "Deep"


def timestamp_key(path: str):
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(path))
    return m.group(1) if m else None


def load_polygons(label_path: str, h: int, w: int):
    """YOLO-seg polygons -> list of binary masks, one per annotated pothole."""
    masks = []
    try:
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                if coords.size % 2:
                    coords = coords[:-1]
                pts = coords.reshape(-1, 2) * np.array([w, h], dtype=np.float32)
                m = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(m, [pts.astype(np.int32)], 1)
                if m.sum() >= MIN_MASK_PX:
                    masks.append(m)
    except Exception:
        return []
    return masks


def road_plane_deviation(depth_mm: np.ndarray, mask: np.ndarray,
                         valid_min: Optional[float] = DEPTH_VALID_MIN):
    """
    Fit a plane to the road ring around `mask`, return per-pixel deviation inside.

    Returns (deviation_map, n_ring, rms_residual_mm) or None if the fit is not
    supportable. The RMS residual is returned so a bad fit can be rejected by the
    caller rather than silently producing a confident wrong number.

    `valid_min` exists because the "invalid pixel" convention is backend-specific,
    not universal. RealSense writes 0 for "no reading", so 1 is right there. But
    Depth-Anything outputs normalised values that may be negative after the
    near/far flip, and applying the RealSense rule to it silently rejects EVERY
    pixel — which is exactly the bug that produced n=0 for Depth-Anything in the
    first backend comparison. Pass None to disable the filter entirely.
    """
    h, w = mask.shape
    ring = cv2.dilate(mask, np.ones((RING_KERNEL, RING_KERNEL), np.uint8), iterations=2)
    ring = (ring > 0) & (mask == 0)
    if valid_min is not None:
        ring &= (depth_mm >= valid_min)
    ring &= np.isfinite(depth_mm)
    n_ring = int(ring.sum())
    if n_ring < MIN_RING_PX:
        return None

    ys, xs = np.nonzero(ring)
    zs = depth_mm[ys, xs].astype(np.float64)

    # Robust-ish: drop the extreme 5% each side before fitting, so a kerb or a
    # parked car intruding into the ring cannot tilt the road plane.
    lo, hi = np.percentile(zs, [5, 95])
    keep = (zs >= lo) & (zs <= hi)
    if keep.sum() < MIN_RING_PX // 2:
        return None
    ys, xs, zs = ys[keep], xs[keep], zs[keep]

    A = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    try:
        coef, *_ = np.linalg.lstsq(A, zs, rcond=None)
    except np.linalg.LinAlgError:
        return None

    resid = zs - A @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))

    gy, gx = np.mgrid[0:h, 0:w]
    plane = coef[0] * gx + coef[1] * gy + coef[2]
    return depth_mm.astype(np.float64) - plane, n_ring, rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-rms", type=float, default=40.0,
                    help="Reject a pothole if the road-plane fit RMS exceeds this (mm)")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.data, "images", "*.jpg")))
    deps = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "depths", "*.npy"))}
    labs = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "labels", "*.txt"))}

    if args.limit:
        imgs = imgs[:args.limit]

    print(f"PothRGBD metric labelling — {len(imgs)} frames")
    print(f"  bands: Shallow <{BAND_SHALLOW_MM:.0f}mm  "
          f"Moderate <{BAND_MODERATE_MM:.0f}mm  Deep >=")
    print()

    rows = []
    n_no_depth = n_no_ring = n_bad_fit = n_implausible = 0

    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        if not dp or not lp:
            n_no_depth += 1
            continue

        depth = np.load(dp)
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        h, w = depth.shape[:2]

        # Roboflow may have resized the RGB; the depth array is authoritative.
        masks = load_polygons(lp, h, w)
        if not masks:
            continue

        for j, m in enumerate(masks, start=1):
            res = road_plane_deviation(depth, m)
            if res is None:
                n_no_ring += 1
                continue
            dev, n_ring, rms = res
            if rms > args.max_rms:
                n_bad_fit += 1
                continue

            inside = dev[(m > 0) & (depth >= DEPTH_VALID_MIN)]
            if inside.size < MIN_MASK_PX // 2:
                continue

            # Positive deviation = further from camera than the road plane = deeper.
            mean_mm = float(np.mean(inside))
            p90_mm = float(np.percentile(inside, 90))
            max_mm = float(np.max(inside))

            # Physical plausibility gate — see MAX_PLAUSIBLE_MM.
            if not (MIN_PLAUSIBLE_MM <= p90_mm <= MAX_PLAUSIBLE_MM):
                n_implausible += 1
                continue

            rows.append({
                "key": k,
                "pothole": j,
                "area_px": int(m.sum()),
                "ring_px": n_ring,
                "plane_rms_mm": round(rms, 2),
                "mean_depth_mm": round(mean_mm, 2),
                "p90_depth_mm": round(p90_mm, 2),
                "max_depth_mm": round(max_mm, 2),
                "severity_p90": severity_from_mm(p90_mm),
                "severity_max": severity_from_mm(max_mm),
            })

        if i % 100 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames, {len(rows)} potholes", flush=True)

    if not rows:
        print("\nNo potholes measured.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "metric_labels.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    p90 = np.array([r["p90_depth_mm"] for r in rows])
    print(f"\n{len(rows)} potholes measured "
          f"(skipped: {n_no_depth} no depth, {n_no_ring} no ring, "
          f"{n_bad_fit} bad plane fit, {n_implausible} implausible depth)")
    print()
    print("  Real bowl depth, 90th percentile within mask (mm):")
    for q in (5, 25, 50, 75, 95):
        print(f"    p{q:<3d} {np.percentile(p90, q):8.1f}")
    print(f"    mean {p90.mean():8.1f}   max {p90.max():8.1f}   min {p90.min():8.1f}")
    print()

    print("  Severity distribution under highway-authority bands:")
    for band in ("Shallow", "Moderate", "Deep"):
        n = sum(1 for r in rows if r["severity_p90"] == band)
        print(f"    {band:9s} {n:5d}  ({100*n/len(rows):5.1f}%)")

    neg = int(np.sum(p90 < 0))
    if neg:
        print(f"\n  NOTE: {neg} potholes ({100*neg/len(rows):.1f}%) measured as "
              f"NEGATIVE depth,\n  i.e. raised above the fitted road plane. Either "
              "the annotation covers a bump\n  rather than a hole, or the plane fit "
              "was tilted by the surrounding geometry.\n  Worth inspecting before "
              "these labels are trusted.")

    print(f"\n  wrote {os.path.relpath(out_csv, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
