"""
Four-way comparison against REAL measured depth, on PothRGBD.

This is the experiment the project has never been able to run. Every previous
comparison (SfS vs Depth-Anything, geometry vs depth) measured two estimators
against each other, which tells you they disagree but never which one is right.
PothRGBD ships Intel RealSense depth in millimetres, so for the first time there
is an arbiter.

Four things are scored against that arbiter:

  1. RealSense           the ground truth itself (road-plane relative, mm)
  2. Depth-Anything-V2   our current backbone — RELATIVE, so correlation only
  3. MoGe                candidate replacement — METRIC, so absolute mm error too
  4. Geometry            boundary curvature etc — never sees depth at all

Two different questions, deliberately kept apart
------------------------------------------------
RANK agreement (Spearman/Pearson): does the estimator order potholes correctly
from shallow to deep? A relative model can win this outright.

ABSOLUTE agreement (mean absolute error in mm): is the number physically right?
Only a metric model can even enter this contest. Depth-Anything cannot, by
construction — its output is normalised per image, so a 2 cm dip and a 20 cm
crater can produce identical maps. That limitation is the whole reason MoGe is
being evaluated.

For geometry we report per-feature correlation against true depth rather than a
single prediction, because the geometry engine outputs shape descriptors, not
millimetres. This directly tests the Phase 1 hypothesis — that boundary shape
carries severity information — against real measurement for the first time.

Usage:
    python scripts/compare_depth_backends_pothrgbd.py --limit 200
    python scripts/compare_depth_backends_pothrgbd.py --limit 200 --no-moge
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

from scripts.pothrgbd_metric_labels import (          # noqa: E402
    DATA_DIR, road_plane_deviation, load_polygons, timestamp_key,
    severity_from_mm, DEPTH_VALID_MIN, MIN_MASK_PX,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")


def bowl_depth_from_map(depth_map: np.ndarray, mask: np.ndarray,
                        max_rms: float = np.inf,
                        valid_min=None):
    """
    Road-plane-relative bowl depth from ANY depth map, in that map's own units.

    Identical treatment for every backend — the same plane fit, the same ring,
    the same percentile. Whatever bias the method has, it applies equally to all
    of them, so the comparison stays fair.
    """
    res = road_plane_deviation(depth_map, mask, valid_min=valid_min)
    if res is None:
        return None
    dev, _n_ring, rms = res
    if rms > max_rms:
        return None
    inside = dev[(mask > 0)]
    inside = inside[np.isfinite(inside)]
    if inside.size < MIN_MASK_PX // 2:
        return None
    return float(np.percentile(inside, 90))


def corr(a, b):
    """Pearson and Spearman, NaN-safe."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    a, b = a[ok], b[ok]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    spear = float(np.corrcoef(ra, rb)[0, 1])
    return pear, spear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-rms", type=float, default=40.0)
    ap.add_argument("--no-moge", action="store_true")
    ap.add_argument("--no-da", action="store_true")
    args = ap.parse_args()

    # ── backends ──
    get_depth_map = None
    if not args.no_da:
        from inference import get_depth_map            # noqa: E402
        print("  Depth-Anything-V2: loaded")

    moge = None
    if not args.no_moge:
        try:
            import moge_backend
            if moge_backend.HAS_MOGE:
                moge = moge_backend
                print(f"  MoGe: {moge_backend.describe()}")
            else:
                print("  MoGe: unavailable (import gate closed)")
        except Exception as e:
            print(f"  MoGe: unavailable ({type(e).__name__}: {e})")

    try:
        from features import extract_curvature_features
        has_geom = True
    except Exception:
        has_geom = False
    print(f"  Geometry: {'loaded' if has_geom else 'unavailable'}")
    print()

    imgs = sorted(glob.glob(os.path.join(args.data, "images", "*.jpg")))
    deps = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "depths", "*.npy"))}
    labs = {timestamp_key(p): p
            for p in glob.glob(os.path.join(args.data, "labels", "*.txt"))}
    if args.limit:
        imgs = imgs[:args.limit]

    rows = []
    for i, ip in enumerate(imgs, start=1):
        k = timestamp_key(ip)
        dp, lp = deps.get(k), labs.get(k)
        if not dp or not lp:
            continue

        gt_depth = np.load(dp)
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        h, w = gt_depth.shape[:2]

        # Every backend is evaluated at the DEPTH array's resolution, so the
        # same mask and the same plane fit apply throughout.
        bgr_r = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA) \
            if bgr.shape[:2] != (h, w) else bgr

        masks = load_polygons(lp, h, w)
        if not masks:
            continue

        da_map = None
        if get_depth_map is not None:
            try:
                da_map = get_depth_map(bgr_r)
                if da_map.shape[:2] != (h, w):
                    da_map = cv2.resize(da_map, (w, h))
                # DA-V2 is inverse-depth-like: larger = nearer. Flip so that
                # "larger = further" matches RealSense, otherwise every
                # correlation below would come out negative for trivial reasons.
                da_map = -da_map.astype(np.float64)
            except Exception:
                da_map = None

        moge_map = None
        if moge is not None:
            try:
                g = moge.get_geometry(bgr_r)
                if g is not None and g.get("depth") is not None:
                    md = np.asarray(g["depth"], dtype=np.float64)
                    if md.shape[:2] != (h, w):
                        md = cv2.resize(md, (w, h))
                    moge_map = md * 1000.0        # metres -> millimetres
            except Exception:
                moge_map = None

        for j, m in enumerate(masks, start=1):
            gt = bowl_depth_from_map(gt_depth, m, args.max_rms,
                                     valid_min=DEPTH_VALID_MIN)
            if gt is None:
                continue

            rec = {"key": k, "pothole": j, "area_px": int(m.sum()),
                   "gt_mm": round(gt, 2), "gt_severity": severity_from_mm(gt)}

            if da_map is not None:
                v = bowl_depth_from_map(da_map, m)
                rec["da_rel"] = round(v, 6) if v is not None else ""
            if moge_map is not None:
                v = bowl_depth_from_map(moge_map, m)
                rec["moge_mm"] = round(v, 2) if v is not None else ""
                if v is not None:
                    rec["moge_severity"] = severity_from_mm(v)
            if has_geom:
                cf = extract_curvature_features(m)
                if cf:
                    for key in ("max_curvature", "mean_curvature",
                                "p90_curvature", "high_curvature_fraction",
                                "contour_elongation"):
                        rec[key] = round(float(cf.get(key, float("nan"))), 6)
            rows.append(rec)

        if i % 25 == 0 or i == len(imgs):
            print(f"    {i}/{len(imgs)} frames, {len(rows)} potholes", flush=True)

    if not rows:
        print("\nNothing measured.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    out = os.path.join(OUT_DIR, "backend_comparison.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        wtr.writerows(rows)

    gt = np.array([r["gt_mm"] for r in rows], float)
    print(f"\n{len(rows)} potholes measured against RealSense ground truth")
    print(f"  true depth: median {np.median(gt):.1f}mm  "
          f"mean {gt.mean():.1f}mm  range {gt.min():.1f}–{gt.max():.1f}mm")

    # ── rank agreement ──
    print(f"\n  RANK AGREEMENT with true depth  (does it order potholes right?)")
    print(f"  {'backend':<22}{'n':>6}{'pearson':>10}{'spearman':>10}")
    print("  " + "-" * 48)

    def num(col):
        return np.array([float(r[col]) if r.get(col) not in ("", None) else np.nan
                         for r in rows], float)

    for col, label in (("da_rel", "Depth-Anything-V2"), ("moge_mm", "MoGe")):
        if col not in keys:
            continue
        v = num(col)
        p, s = corr(gt, v)
        print(f"  {label:<22}{int(np.isfinite(v).sum()):>6}{p:>10.3f}{s:>10.3f}")

    if has_geom:
        print(f"\n  {'geometry feature':<22}{'n':>6}{'pearson':>10}{'spearman':>10}")
        print("  " + "-" * 48)
        for col in ("max_curvature", "mean_curvature", "p90_curvature",
                    "high_curvature_fraction", "contour_elongation"):
            if col not in keys:
                continue
            v = num(col)
            p, s = corr(gt, v)
            print(f"  {col:<22}{int(np.isfinite(v).sum()):>6}{p:>10.3f}{s:>10.3f}")

    # ── absolute agreement — metric backends only ──
    if "moge_mm" in keys:
        v = num("moge_mm")
        ok = np.isfinite(v)
        if ok.sum() >= 3:
            err = v[ok] - gt[ok]
            print(f"\n  ABSOLUTE AGREEMENT in millimetres  (MoGe only — "
                  f"Depth-Anything cannot enter,\n  its output has no scale by "
                  f"construction)")
            print(f"    n              {int(ok.sum())}")
            print(f"    mean abs error {np.mean(np.abs(err)):8.1f} mm")
            print(f"    median abs err {np.median(np.abs(err)):8.1f} mm")
            print(f"    bias (signed)  {np.mean(err):+8.1f} mm")
            within = lambda t: 100.0 * np.mean(np.abs(err) <= t)   # noqa: E731
            print(f"    within  10mm   {within(10):5.1f}%")
            print(f"    within  25mm   {within(25):5.1f}%")

            agree = sum(1 for r in rows
                        if r.get("moge_severity") and
                        r["moge_severity"] == r["gt_severity"])
            n_sev = sum(1 for r in rows if r.get("moge_severity"))
            if n_sev:
                print(f"\n    3-class severity agreement: {agree}/{n_sev} "
                      f"({100*agree/n_sev:.1f}%) using <25 / 25–50 / >50 mm bands")

    print(f"\n  wrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
