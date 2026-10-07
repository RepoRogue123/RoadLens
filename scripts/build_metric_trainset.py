"""
Metric training set: MoGe features that keep their millimetres, under three masks.

Why a second trainset
---------------------
`build_pothrgbd_trainset.py` fed MoGe depth through `features.py`, which min-max
normalises every map first — so the "MoGe" columns lost their units and scored
within 0.3 mm of Depth-Anything. This builder uses `metric_features.py`, which
measures bowl depth below a fitted road plane in millimetres, metric area from
3-D points, and scale-free ratios.

Three mask sources per measured pothole
---------------------------------------
    gt     the human polygon            (the training condition)
    yolo   the production YOLO mask     (the deployed condition)
    sam2   YOLO refined by SAM 2        (the proposed deployed condition)

The label is always computed from RealSense depth under the HUMAN polygon, so
`yolo` and `sam2` rows answer "how good is the estimate when the mask comes from
our own segmenter?" — the number that matters in production. YOLO masks are
matched to polygons by IoU; the IoU is recorded, which also settles the open
SAM 2 question (does refinement move masks toward the human outline?).

MoGe outputs are cached per frame under archive/moge_cache/ (gitignored), so
features can be re-engineered without re-running a 370M-parameter model.

Usage:
    python scripts/build_metric_trainset.py --limit 20     # smoke test
    python scripts/build_metric_trainset.py                # full run (~30 min on an RTX 4070)
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

from metric_features import (                                     # noqa: E402
    FEATURE_COLS, SHAPE_COLS, extract_metric_features, shape_features,
)
from scripts.pothrgbd_metric_labels import (                       # noqa: E402
    DATA_DIR, load_polygons, timestamp_key, road_plane_deviation,
    severity_from_mm, DEPTH_VALID_MIN, MIN_MASK_PX,
    MAX_PLAUSIBLE_MM, MIN_PLAUSIBLE_MM,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
CACHE_DIR = os.path.join(PROJECT_DIR, "archive", "moge_cache")
MATCH_IOU = 0.10          # below this a YOLO mask is not "the same pothole"


def iou(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union else 0.0


def moge_for(key, bgr, moge):
    """MoGe depth/intrinsics/validity/normals for a frame, cached to disk."""
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {"depth": z["depth"].astype(np.float32), "intrinsics": z["intrinsics"],
                "mask": z["mask"], "normal": z["normal"].astype(np.float32)}
    g = moge.get_geometry(bgr)
    if g is None:
        return None
    out = {"depth": np.asarray(g["depth"], np.float32),
           "intrinsics": np.asarray(g["intrinsics"], np.float32),
           "mask": (np.asarray(g["mask"]) > 0.5).astype(np.uint8),
           "normal": np.asarray(g["normal"], np.float32)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, depth=out["depth"].astype(np.float16), intrinsics=out["intrinsics"],
                        mask=out["mask"], normal=out["normal"].astype(np.float16))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-rms", type=float, default=40.0)
    ap.add_argument("--no-sam2", action="store_true")
    args = ap.parse_args()

    import moge_backend
    import segmentation
    if not moge_backend.HAS_MOGE:
        sys.exit("MoGe unavailable — this trainset is meaningless without it.")
    refiner = None
    if not args.no_sam2:
        import mask_refine
        refiner = mask_refine if mask_refine.HAS_SAM2 else None
        print(f"  SAM 2: {'available' if refiner else 'UNAVAILABLE — sam2 rows skipped'}")

    imgs = sorted(glob.glob(os.path.join(args.data, "images", "*.jpg")))
    deps = {timestamp_key(p): p for p in glob.glob(os.path.join(args.data, "depths", "*.npy"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(args.data, "labels", "*.txt"))}
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"\nBuilding metric trainset from {len(imgs)} frames\n")

    rows, n_matched, n_gt = [], 0, 0
    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        bgr = cv2.imread(ip)
        if not dp or not lp or bgr is None:
            continue
        gt_depth = np.load(dp)
        h, w = gt_depth.shape[:2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        polys = load_polygons(lp, h, w)
        if not polys:
            continue

        g = moge_for(k, bgr, moge_backend)
        if g is None:
            continue
        yolo = segmentation._extract_binary_masks(bgr, segmentation.MODEL)

        for j, gm in enumerate(polys, start=1):
            res = road_plane_deviation(gt_depth, gm, valid_min=DEPTH_VALID_MIN)
            if res is None:
                continue
            dev, _n, rms = res
            if rms > args.max_rms:
                continue
            inside = dev[(gm > 0) & (gt_depth >= DEPTH_VALID_MIN)]
            if inside.size < MIN_MASK_PX // 2:
                continue
            gt_mm = float(np.percentile(inside, 90))
            if not (MIN_PLAUSIBLE_MM <= gt_mm <= MAX_PLAUSIBLE_MM):
                continue
            n_gt += 1

            sources = [("gt", gm, 1.0)]
            if yolo:
                ious = [iou(ym, gm) for ym in yolo]
                b = int(np.argmax(ious))
                if ious[b] >= MATCH_IOU:
                    n_matched += 1
                    ym = yolo[b]
                    sources.append(("yolo", ym, ious[b]))
                    if refiner is not None:
                        rm = refiner.refine_mask(rgb, ym)
                        sources.append(("sam2", rm, iou(rm, gm)))

            for src, m, m_iou in sources:
                mf = extract_metric_features(m, g["depth"], g["intrinsics"],
                                             normal=g["normal"], valid=g["mask"])
                sf = shape_features(m)
                if mf is None or sf is None:
                    continue
                rec = {"key": k, "pothole": j, "mask_source": src,
                       "mask_iou_vs_gt": round(m_iou, 4),
                       "gt_mm": round(gt_mm, 3), "gt_severity": severity_from_mm(gt_mm)}
                rec.update({c: round(v, 6) for c, v in mf.items()})
                rec.update({c: round(v, 6) for c, v in sf.items()})
                rows.append(rec)

        if i % 50 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames  {n_gt} measured potholes  "
                  f"{n_matched} matched by YOLO  {len(rows)} rows", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "trainset_metric.csv")
    cols = ["key", "pothole", "mask_source", "mask_iou_vs_gt", "gt_mm", "gt_severity"]
    cols += FEATURE_COLS + SHAPE_COLS
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n{n_gt} measured potholes; YOLO matched {n_matched} ({100*n_matched/max(n_gt,1):.1f}%)")
    for src in ("gt", "yolo", "sam2"):
        r = [x for x in rows if x["mask_source"] == src]
        if r:
            ious = np.array([x["mask_iou_vs_gt"] for x in r])
            print(f"  {src:5s} rows {len(r):5d}   IoU vs human polygon: mean {ious.mean():.3f}  median {np.median(ious):.3f}")
    print(f"\n  wrote {os.path.relpath(out, PROJECT_DIR)} — SPLIT ON 'key'")


if __name__ == "__main__":
    main()
