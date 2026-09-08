"""
Build a training set whose labels are MEASURED, not inferred.

Why this exists
---------------
Every model in this project was trained on labels derived from Depth-Anything-V2
(KMeans over `max_depth` and area). Measured against RealSense ground truth,
DA-V2 scores **Pearson 0.019 / Spearman 0.047** — no correlation with real
pothole depth at all. So the pseudo-labels encode nothing physical, and the
99.7% "All Features" accuracy was the system reproducing a meaningless signal.

PothRGBD gives real millimetres. This script extracts the same feature vectors
the pipeline already uses, computed under BOTH depth backends, and joins them to
the measured depth so the models can finally be trained against reality.

What comes out
--------------
One wide CSV. Every one of the 39 pipeline features is emitted twice, once per
backend, prefixed `da_` and `moge_`. Duplicating the mask-only features (the ten
curvature columns, the bounding-box geometry) across both prefixes is deliberate:
it keeps the two variants selectable by prefix alone, so the trainer never has to
know which features happen to be depth-independent.

    key, pothole            frame id and index — SPLIT ON `key`, never on row
    gt_mm, gt_severity      measured ground truth
    area_px, log_area       log_area is currently the single strongest predictor
                            (r = +0.511) and is the baseline any feature set must beat
    da_<39>                 the 39 features under Depth-Anything-V2
    moge_<39>               the same 39 under MoGe-3 metric depth

Splitting must be BY FRAME. Two potholes in one photograph share lighting, camera
pose, road surface and operator; splitting by pothole leaks all of that across
the boundary and inflates the score.

Usage:
    python scripts/build_pothrgbd_trainset.py --limit 50     # smoke test
    python scripts/build_pothrgbd_trainset.py                # full run
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from features import (                                    # noqa: E402
    extract_features_extended, extract_all_geometry_features,
)
from inference import FEATURE_COLS_20, GEOMETRY_FEATURE_COLS   # noqa: E402
from scripts.pothrgbd_metric_labels import (                # noqa: E402
    DATA_DIR, load_polygons, timestamp_key, road_plane_deviation,
    severity_from_mm, DEPTH_VALID_MIN, MIN_MASK_PX,
    MAX_PLAUSIBLE_MM, MIN_PLAUSIBLE_MM,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
ALL_COLS = FEATURE_COLS_20 + GEOMETRY_FEATURE_COLS          # 39


def features_for(mask: np.ndarray, depth_map: np.ndarray, image_rgb):
    """
    The pipeline's own 39-feature vector, as a dict.

    Reuses extract_features_extended and extract_all_geometry_features rather
    than reimplementing, so anything measured here is exactly what the live
    pipeline would compute for the same inputs.
    """
    out = {c: 0.0 for c in ALL_COLS}

    ext = extract_features_extended(mask, depth_map)
    if ext is None:
        return None
    for c in FEATURE_COLS_20:
        v = ext.get(c)
        if v is not None:
            out[c] = float(v)

    geo = extract_all_geometry_features(mask, depth_map, image_rgb)
    if geo:
        for c in GEOMETRY_FEATURE_COLS:
            v = geo.get(c)
            if isinstance(v, (int, float)) and np.isfinite(v):
                out[c] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-rms", type=float, default=40.0)
    ap.add_argument("--no-moge", action="store_true")
    args = ap.parse_args()

    from inference import get_depth_map
    print("  Depth-Anything-V2: loaded")

    moge = None
    if not args.no_moge:
        import moge_backend
        if moge_backend.HAS_MOGE:
            moge = moge_backend
            print(f"  MoGe: {moge_backend.describe()}")
        else:
            print("  MoGe: UNAVAILABLE — moge_ columns will be empty")

    imgs = sorted(glob.glob(os.path.join(args.data, "images", "*.jpg")))
    deps = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "depths", "*.npy"))}
    labs = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "labels", "*.txt"))}
    if args.limit:
        imgs = imgs[:args.limit]

    print(f"\nBuilding real-label training set from {len(imgs)} frames\n")

    rows = []
    n_skip = 0
    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        if not dp or not lp:
            n_skip += 1
            continue

        gt_depth = np.load(dp)
        bgr = cv2.imread(ip)
        if bgr is None:
            n_skip += 1
            continue
        h, w = gt_depth.shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        masks = load_polygons(lp, h, w)
        if not masks:
            continue

        # Depth maps, once per frame rather than once per pothole.
        try:
            da_map = get_depth_map(bgr)
            if da_map.shape[:2] != (h, w):
                da_map = cv2.resize(da_map, (w, h))
            da_map = da_map.astype(np.float32)
        except Exception:
            da_map = None

        moge_map = None
        if moge is not None:
            g = moge.get_geometry(bgr)
            if g is not None and g.get("depth") is not None:
                md = np.asarray(g["depth"], dtype=np.float32)
                if md.shape[:2] != (h, w):
                    md = cv2.resize(md, (w, h))
                moge_map = md * 1000.0                  # metres -> millimetres

        for j, m in enumerate(masks, start=1):
            # ── measured ground truth, identical logic to metric_labels.py ──
            res = road_plane_deviation(gt_depth, m, valid_min=DEPTH_VALID_MIN)
            if res is None:
                continue
            dev, _n_ring, rms = res
            if rms > args.max_rms:
                continue
            inside = dev[(m > 0) & (gt_depth >= DEPTH_VALID_MIN)]
            if inside.size < MIN_MASK_PX // 2:
                continue
            gt_mm = float(np.percentile(inside, 90))
            if not (MIN_PLAUSIBLE_MM <= gt_mm <= MAX_PLAUSIBLE_MM):
                continue

            area = int(m.sum())
            rec = {
                "key": k, "pothole": j,
                "gt_mm": round(gt_mm, 3),
                "gt_severity": severity_from_mm(gt_mm),
                "area_px": area,
                "log_area": round(float(np.log(max(area, 1))), 6),
            }

            ok = True
            for prefix, dmap in (("da", da_map), ("moge", moge_map)):
                if dmap is None:
                    ok = False
                    break
                f = features_for(m, dmap, rgb)
                if f is None:
                    ok = False
                    break
                for c, v in f.items():
                    rec[f"{prefix}_{c}"] = round(v, 6)
            if not ok:
                continue

            rows.append(rec)

        if i % 50 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames, {len(rows)} potholes", flush=True)

    if not rows:
        print("\nNothing built.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "trainset_real_labels.csv")
    cols = ["key", "pothole", "gt_mm", "gt_severity", "area_px", "log_area"]
    cols += [f"da_{c}" for c in ALL_COLS] + [f"moge_{c}" for c in ALL_COLS]
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        wtr.writerows(rows)

    gt = np.array([r["gt_mm"] for r in rows])
    n_frames = len({r["key"] for r in rows})
    print(f"\n{len(rows)} potholes from {n_frames} frames  (skipped {n_skip})")
    print(f"  gt depth: median {np.median(gt):.1f}mm  mean {gt.mean():.1f}mm  "
          f"range {gt.min():.1f}-{gt.max():.1f}mm")
    for band in ("Shallow", "Moderate", "Deep"):
        n = sum(1 for r in rows if r["gt_severity"] == band)
        print(f"    {band:9s} {n:5d}  ({100*n/len(rows):5.1f}%)")
    print(f"\n  BASELINES any model must beat:")
    print(f"    predict median   MAE {np.mean(np.abs(gt - np.median(gt))):6.2f} mm")
    la = np.array([r["log_area"] for r in rows])
    A = np.column_stack([la, np.ones_like(la)])
    c, *_ = np.linalg.lstsq(A, gt, rcond=None)
    print(f"    log_area alone   MAE {np.mean(np.abs(A @ c - gt)):6.2f} mm")
    print(f"\n  wrote {os.path.relpath(out, PROJECT_DIR)}")
    print(f"  columns: {len(cols)}  — SPLIT ON 'key', never on row index")


if __name__ == "__main__":
    main()
