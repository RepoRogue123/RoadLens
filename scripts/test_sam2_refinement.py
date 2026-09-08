"""
M4.1 verification — does SAM 2 refinement move YOLO masks CLOSER to human truth?

Why this test exists
--------------------
On a first look, SAM 2 shrinks our YOLO masks by 25-30%. That is a large change,
and "different" is not "better". Every previous mask change in this project was
adopted on the assumption that a sharper boundary must help; none was measured.

PothRGBD carries human-drawn segmentation polygons, so the question is directly
answerable: run YOLO on the frame, refine with SAM 2, and score BOTH against the
human annotation. If refinement helps, IoU against the human polygon goes up.

Also reported, because the mask is not the end goal:

  * boundary-fit quality — RMS of the road-plane fit around each mask, in mm.
    A mask that leaks onto the road drags road pixels into the interior and
    pothole pixels into the ring, so a cleaner mask should lower this.
  * measured bowl depth — how much the refinement moves the millimetre reading.

Matching: a predicted mask is paired to the ground-truth polygon it overlaps
most, and unmatched predictions are counted rather than dropped, so the score
cannot be inflated by silently discarding bad detections.

Usage:
    python scripts/test_sam2_refinement.py --limit 150
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

from scripts.pothrgbd_metric_labels import (               # noqa: E402
    DATA_DIR, load_polygons, timestamp_key, road_plane_deviation,
    DEPTH_VALID_MIN, MIN_MASK_PX,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def measure(depth, mask, max_rms=1e9):
    """(bowl_depth_mm, plane_rms_mm) for a mask, or (nan, nan)."""
    res = road_plane_deviation(depth, mask, valid_min=DEPTH_VALID_MIN)
    if res is None:
        return float("nan"), float("nan")
    dev, _n, rms = res
    inside = dev[(mask > 0) & (depth >= DEPTH_VALID_MIN)]
    if inside.size < MIN_MASK_PX // 2:
        return float("nan"), rms
    return float(np.percentile(inside, 90)), rms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    import mask_refine
    if not mask_refine.HAS_SAM2:
        print("SAM 2 unavailable — nothing to test")
        return
    print(f"  {mask_refine.describe()}")

    from segmentation import get_all_masks
    print("  YOLOv8-seg: loaded\n")

    imgs = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.jpg")))[:args.limit]
    deps = {timestamp_key(p): p
            for p in glob.glob(os.path.join(DATA_DIR, "depths", "*.npy"))}
    labs = {timestamp_key(p): p
            for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}

    rows = []
    n_unmatched = 0
    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        if not dp or not lp:
            continue
        depth = np.load(dp)
        h, w = depth.shape[:2]
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gts = load_polygons(lp, h, w)
        if not gts:
            continue

        try:
            preds = get_all_masks(ip)
        except Exception:
            continue
        if not preds:
            continue

        for pm in preds:
            if pm.shape[:2] != (h, w):
                pm = cv2.resize(pm.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST)
            # pair with the ground-truth polygon it overlaps most
            ious = [iou(pm, g) for g in gts]
            best = int(np.argmax(ious))
            if ious[best] <= 0.0:
                n_unmatched += 1
                continue
            gt = gts[best]

            rm = mask_refine.refine_mask(rgb, pm)

            d_c, rms_c = measure(depth, pm)
            d_r, rms_r = measure(depth, rm)
            d_g, rms_g = measure(depth, gt)

            rows.append({
                "key": k,
                "iou_coarse": round(ious[best], 4),
                "iou_refined": round(iou(rm, gt), 4),
                "area_coarse": int((pm > 0).sum()),
                "area_refined": int((rm > 0).sum()),
                "area_gt": int((gt > 0).sum()),
                "rms_coarse": round(rms_c, 3),
                "rms_refined": round(rms_r, 3),
                "rms_gt": round(rms_g, 3),
                "depth_coarse": round(d_c, 3),
                "depth_refined": round(d_r, 3),
                "depth_gt": round(d_g, 3),
            })

        if i % 25 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames, {len(rows)} matched masks",
                  flush=True)

    if not rows:
        print("\nNothing matched.")
        return

    def col(n):
        return np.array([r[n] for r in rows], dtype=float)

    ic, ir = col("iou_coarse"), col("iou_refined")
    better = int((ir > ic).sum())
    worse = int((ir < ic).sum())

    print(f"\n{len(rows)} matched masks, {n_unmatched} predictions with no "
          f"overlapping ground truth")
    print(f"\n  DOES REFINEMENT MOVE MASKS TOWARD HUMAN TRUTH?")
    print(f"    mean IoU vs human   coarse {ic.mean():.4f}   "
          f"refined {ir.mean():.4f}   delta {ir.mean()-ic.mean():+.4f}")
    print(f"    median IoU          coarse {np.median(ic):.4f}   "
          f"refined {np.median(ir):.4f}")
    print(f"    improved {better}/{len(rows)} ({100*better/len(rows):.1f}%)   "
          f"worsened {worse} ({100*worse/len(rows):.1f}%)")

    ac, ar, ag = col("area_coarse"), col("area_refined"), col("area_gt")
    print(f"\n  AREA vs human truth")
    print(f"    coarse/gt  median ratio {np.median(ac/np.maximum(ag,1)):.3f}")
    print(f"    refined/gt median ratio {np.median(ar/np.maximum(ag,1)):.3f}")
    print("    (1.0 is perfect; >1 over-segments onto the road, <1 under-covers)")

    rc, rr, rg = col("rms_coarse"), col("rms_refined"), col("rms_gt")
    ok = np.isfinite(rc) & np.isfinite(rr)
    print(f"\n  ROAD-PLANE FIT QUALITY (mm RMS, lower is better)")
    print(f"    coarse  {np.nanmedian(rc):6.2f}    refined {np.nanmedian(rr):6.2f}"
          f"    human {np.nanmedian(rg):6.2f}")

    dc, dr, dg = col("depth_coarse"), col("depth_refined"), col("depth_gt")
    ok2 = np.isfinite(dc) & np.isfinite(dr) & np.isfinite(dg)
    if ok2.sum() > 5:
        print(f"\n  MEASURED DEPTH ERROR vs human-mask depth (mm)")
        print(f"    coarse  MAE {np.mean(np.abs(dc[ok2]-dg[ok2])):6.2f}")
        print(f"    refined MAE {np.mean(np.abs(dr[ok2]-dg[ok2])):6.2f}")

    print(f"\n  gate: {mask_refine.stats()}")

    verdict = ir.mean() - ic.mean()
    print()
    if verdict > 0.01:
        print(f"  VERDICT: refinement helps (+{verdict:.4f} mean IoU). Worth adopting.")
    elif verdict < -0.01:
        print(f"  VERDICT: refinement HURTS ({verdict:+.4f} mean IoU). Do not adopt —")
        print("  SAM 2 is tightening onto something other than the annotated pothole.")
    else:
        print(f"  VERDICT: no meaningful change ({verdict:+.4f} mean IoU). Not worth")
        print("  the latency; the boundary was not the limiting factor.")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "sam2_refinement_test.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\n  wrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
