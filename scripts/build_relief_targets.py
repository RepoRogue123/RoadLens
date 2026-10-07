"""
Inspect a measured-depth dataset, then turn it into relief targets and pothole labels
with the same protocol as PothRGBD.

Inspection comes first, and can be run alone (--inspect-only), because two datasets in
this project already turned out not to hold what their descriptions promised. Printed and
written to ml_results/depth_sources/<source>_inspection.json:

    resolution, share of pixels with a reading, camera distance and tilt,
    flat-road noise floor  (spread of the road around a plane: what "0 mm" looks like),
    how many potholes, and how deep.

Then, per frame, at the 640-pixel scale the served models work at:

    relief target   depth minus one robust road plane, mm, positive = deeper
                    (scripts/train_relief_depth.relief_target with the exact, inverse-depth
                    plane: the linear one leaves 25 mm of false relief on tilted close-ups)
    pothole label   90th percentile of the pothole's depth below a plane fitted on the ring
                    of road around it (metric_features.road_plane_deviation, unchanged)

Two depths are recorded for every pothole, because they are not the same thing:

    gt_mm     the PothRGBD protocol: depth measured ALONG THE CAMERA AXIS. This is what
              every model in the project was trained to predict.
    perp_mm   depth measured PERPENDICULAR to the road, from a 3D plane (only where the
              source has a point cloud). This is what a ruler would read.

A camera looking straight down gives the same number for both. A tilted camera reads
more along its axis than the ruler does, by 1 / cos(tilt).

Output: archive/relief_targets/<source>/<key>.npz  (gitignored) and
        ml_results/depth_sources/<source>_potholes.csv

Usage:
    python scripts/build_relief_targets.py --source fan [--inspect-only]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from metric_features import MIN_INSIDE_PX, TRAIN_LONG_SIDE, road_plane_deviation  # noqa: E402
from scripts.depth_sources import PRESENT, READERS                                # noqa: E402
from scripts.train_relief_depth import relief_target                              # noqa: E402

CACHE_DIR = os.path.join(PROJECT_DIR, "archive", "relief_targets")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "depth_sources")
MAX_PLAUSIBLE_MM, MIN_PLAUSIBLE_MM = 250.0, -10.0            # as scripts/pothrgbd_metric_labels.py


def to_scale(frame):
    """Resize a frame to the served scale. Depth and labels by nearest pixel: never blended."""
    h, w = frame["bgr"].shape[:2]
    s = TRAIN_LONG_SIDE / max(h, w)
    if s >= 1:
        return frame
    size = (int(round(w * s)), int(round(h * s)))
    out = dict(frame)
    if frame.get("K") is not None:
        out["K"] = frame["K"].copy()
        out["K"][:2] *= s
    out["bgr"] = cv2.resize(frame["bgr"], size, interpolation=cv2.INTER_AREA)
    for k in ("depth_mm", "mask", "xyz_mm"):
        if frame.get(k) is not None:
            out[k] = cv2.resize(frame[k], size, interpolation=cv2.INTER_NEAREST)
    return out


DEPRESSION_MM = 15.0          # a region this far below the road counts as a depression
BLOCK = 4                     # px; readings are pooled in blocks first (RSRD: ~16% of pixels measured)
MIN_BLOCKS = 6


def depressions(relief, valid):
    """
    Pothole-like regions found in the MEASURED relief, for datasets with no outlines (RSRD).

    Readings are averaged in 4x4-px blocks (most pixels have none), blocks lying more than
    15 mm below the road plane are joined into regions, and regions of at least 6 blocks are
    kept. The result plays the part of the human outline in scoring: the network's prediction
    is then read inside it with the label protocol. These regions are defined by the truth, so
    they say where to look, never what to predict.
    """
    h, w = relief.shape
    hb, wb = h // BLOCK, w // BLOCK
    r = np.where(valid, relief, 0)[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3))
    n = valid[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3))
    mean = np.where(n > 0, r / np.maximum(n, 1), 0)
    deep = ((mean > DEPRESSION_MM) & (n > 0)).astype(np.uint8)
    deep = cv2.morphologyEx(deep, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    k, comp = cv2.connectedComponents(deep)
    out = np.zeros((h, w), np.uint8)
    for c in range(1, k):
        m = comp == c
        if m.sum() >= MIN_BLOCKS:
            big = cv2.resize(m.astype(np.uint8), (wb * BLOCK, hb * BLOCK), interpolation=cv2.INTER_NEAREST)
            out[:hb * BLOCK, :wb * BLOCK] |= big
    return out


def relief_target_perp(depth_mm, K, polys):
    """
    (relief_mm, valid, pothole_mask, road_spread) with relief measured PERPENDICULAR to the road.

    For cameras that look ALONG the road rather than down at it. PothRGBD and Fan look nearly
    straight down, where a dip measured along the line of sight equals the dip below the road.
    RSRD's line of sight meets the road at about 16 degrees, so the same dip reads about 3.6
    times deeper along the line of sight, and so does the noise (road spread 36 mm along the
    line of sight, 6.4 mm perpendicular). Here every reading is placed in 3D with the camera
    intrinsics K, a gently curved surface (quadratic in the road's two horizontal directions)
    is fitted to the road (trimmed), and relief is the distance below it.

    Why curved: over 2-9 m a road is rarely one plane. With a plane, the far edge of almost
    every RSRD frame read 20-40 mm "below the road" and was flagged as a depression; the
    label protocol only ever looks at a ring around a pothole, so a target that keeps the
    road's large-scale shape asks the network for something the labels never measure.
    """
    h, w = depth_mm.shape
    ok = depth_mm > 0
    pot = np.zeros((h, w), np.uint8)
    for m in polys:
        pot |= m
    road = ok & (cv2.dilate(pot, np.ones((15, 15), np.uint8)) == 0)
    if road.sum() < 2000:
        return None
    gy, gx = np.mgrid[0:h, 0:w]
    Z = depth_mm.astype(np.float64)
    X = (gx - K[0, 2]) * Z / K[0, 0]
    Y = (gy - K[1, 2]) * Z / K[1, 1]
    def design(x, z):                        # road height Y as a quadratic in X (across) and Z (ahead)
        return np.column_stack([x, z, np.ones(len(x)), x * x, x * z, z * z])
    xr, zr = X[road], Z[road]
    xs, zs = xr / 1000.0, zr / 1000.0        # metres, for a well-conditioned fit
    A = design(xs, zs)
    keep = np.ones(len(A), bool)
    for _ in range(4):                       # camera Y points down, so larger Y = lower
        c, *_ = np.linalg.lstsq(A[keep], Y[road][keep], rcond=None)
        r = Y[road] - A @ c
        keep = np.abs(r) < 2.5 * r[keep].std()
    n_y = 1.0 / np.sqrt(1 + (c[0] / 1000) ** 2 + (c[1] / 1000) ** 2)   # cosine between road normal and camera Y
    surface = (design(X.ravel() / 1000.0, Z.ravel() / 1000.0) @ c).reshape(h, w)
    relief = (Y - surface) * n_y                          # positive = below the road
    valid = ok & (np.abs(relief) < 300.0)
    return relief.astype(np.float32), valid, pot, float(r[keep].std() * n_y)


def plane_3d(points):
    """Trimmed least-squares plane through Nx3 points: (centre, unit normal facing away from the camera, rms)."""
    keep = np.ones(len(points), bool)
    for _ in range(3):
        c = points[keep].mean(0)
        _u, _s, vt = np.linalg.svd(points[keep] - c, full_matrices=False)
        n = vt[2] if vt[2][2] > 0 else -vt[2]
        d = (points - c) @ n
        keep = np.abs(d) < 2.5 * d[keep].std()
    return c, n, float(d[keep].std())


def pothole_rows(frame):
    """One row per labelled pothole: both depths, the ring's noise, the camera's tilt and distance."""
    if frame["mask"] is None:
        return []
    valid = frame["depth_mm"] > 0
    # Perpendicular relief when the camera looks along the road (relief_target_perp): the label
    # protocol is then applied to it, exactly as it is applied to a predicted relief map.
    depth = frame["relief_map"] if frame.get("relief_map") is not None else frame["depth_mm"]
    n, comp = cv2.connectedComponents(frame["mask"])
    rows = []
    for c in range(1, n):
        m = (comp == c).astype(np.uint8)
        inside = (m > 0) & valid
        if int(inside.sum()) < MIN_INSIDE_PX:
            continue
        res = road_plane_deviation(depth, m, valid & ((frame["mask"] == 0) | (m > 0)))
        if res is None:
            continue
        dev, ring, rms = res
        gt = float(np.percentile(dev[inside], 90))
        if not MIN_PLAUSIBLE_MM <= gt <= MAX_PLAUSIBLE_MM:
            continue
        row = {"key": frame["key"], "group": frame["group"], "pothole": c, "gt_mm": gt, "px": int(m.sum()),
               "ring_rms_mm": rms, "camera_mm": float(np.median(frame["depth_mm"][inside])), "flagged": frame["flagged"],
               "perp_mm": np.nan, "tilt_deg": np.nan}
        if frame.get("xyz_mm") is not None:
            centre, normal, _ = plane_3d(frame["xyz_mm"][ring].astype(np.float64))
            row["perp_mm"] = float(np.percentile((frame["xyz_mm"][inside].astype(np.float64) - centre) @ normal, 90))
            row["tilt_deg"] = float(np.degrees(np.arccos(np.clip(normal[2], -1, 1))))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=sorted(READERS))
    ap.add_argument("--inspect-only", action="store_true", help="report, write nothing but the report")
    args = ap.parse_args()
    if not PRESENT[args.source]:
        sys.exit(f"{args.source} is not on disk (see scripts/depth_sources.py for where it goes)")

    cache = os.path.join(CACHE_DIR, args.source)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.inspect_only:
        os.makedirs(cache, exist_ok=True)

    rows, frames, sizes = [], [], set()
    for raw in READERS[args.source]():
        sizes.add(raw["bgr"].shape[:2])
        f = to_scale(raw)
        polys = []
        if f["mask"] is not None:
            n, comp = cv2.connectedComponents(f["mask"])
            polys = [(comp == c).astype(np.uint8) for c in range(1, n)]
        target = (lambda p: relief_target_perp(f["depth_mm"], f["K"], p)) if f.get("K") is not None \
            else (lambda p: relief_target(f["depth_mm"], p, exact=True))
        t = target(polys)
        if f["mask"] is None and t is not None:
            # No outlines: find depressions in the measured relief, then refit the road plane
            # without them so they do not drag it down.
            f["mask"] = depressions(t[0], t[1])
            n, comp = cv2.connectedComponents(f["mask"])
            polys = [(comp == c).astype(np.uint8) for c in range(1, n)]
            t = target(polys) or t
        if f.get("K") is not None and t is not None:
            f["relief_map"] = np.where(t[1], t[0], 0).astype(np.float32)   # labels read from perpendicular relief
        pr = pothole_rows(f)
        rows += pr
        frames.append({"key": f["key"], "group": f["group"], "valid_share": float((f["depth_mm"] > 0).mean()),
                       "road_std_mm": t[3] if t else np.nan, "potholes": len(pr),
                       "relief_p99_mm": float(np.percentile(t[0][t[1]], 99)) if t else np.nan})
        if not args.inspect_only and t is not None:
            np.savez_compressed(os.path.join(cache, f["key"] + ".npz"), bgr=f["bgr"], depth_mm=f["depth_mm"],
                                mask=f["mask"] if f["mask"] is not None else np.zeros(f["depth_mm"].shape, np.uint8),
                                relief=t[0], valid=t[1], road_std=t[3], group=f["group"], flagged=f["flagged"])
        if len(frames) % 250 == 0:
            print(f"  {len(frames)} frames", flush=True)

    fr, d = pd.DataFrame(frames), pd.DataFrame(rows)
    report = {
        "source": args.source, "frames": int(len(fr)), "native_sizes_hw": sorted(map(list, sizes)),
        "valid_share_median": float(fr.valid_share.median()),
        "road_std_mm": {"median": float(fr.road_std_mm.median()), "p90": float(fr.road_std_mm.quantile(0.9))},
        "frames_with_relief_over_15mm": int((fr.relief_p99_mm > 15).sum()),
        "potholes": int(len(d)),
    }
    print(f"\n== {args.source}: {len(fr)} frames, native size {sorted(sizes)}, scored at long side {TRAIN_LONG_SIDE}")
    print(f"   pixels with a reading: {fr.valid_share.median():.1%} (median frame)")
    print(f"   flat-road noise floor: {fr.road_std_mm.median():.2f} mm (median frame), {fr.road_std_mm.quantile(0.9):.2f} mm (90th percentile)")
    print(f"   frames whose deepest 1% lies more than 15 mm below the road: {report['frames_with_relief_over_15mm']}")
    if len(d):
        q = lambda s: [float(v) for v in np.percentile(s.dropna(), [10, 50, 90])]      # noqa: E731
        report.update({"gt_mm_p10_50_90": q(d.gt_mm), "perp_mm_p10_50_90": q(d.perp_mm) if d.perp_mm.notna().any() else None,
                       "camera_mm_p10_50_90": q(d.camera_mm), "tilt_deg_p10_50_90": q(d.tilt_deg) if d.tilt_deg.notna().any() else None,
                       "ring_rms_mm_median": float(d.ring_rms_mm.median()), "flagged_potholes": int(d.flagged.sum()),
                       "bands_gt": {b: int(n) for b, n in zip(("Shallow", "Moderate", "Deep"),
                                                             np.histogram(d.gt_mm, [-np.inf, 25, 50, np.inf])[0])}})
        print(f"   {len(d)} labelled potholes ({int(d.flagged.sum())} in frames the authors set aside), by group: {d.group.value_counts().to_dict()}")
        print(f"   depth along the camera axis (the PothRGBD protocol), mm: p10 {report['gt_mm_p10_50_90'][0]:.1f}, median {report['gt_mm_p10_50_90'][1]:.1f}, p90 {report['gt_mm_p10_50_90'][2]:.1f}"
              f"   -> {report['bands_gt']}")
        if report["perp_mm_p10_50_90"]:
            ratio = (d.gt_mm / d.perp_mm)[d.perp_mm > 5]
            report["axis_over_perpendicular_median"] = float(ratio.median())
            print(f"   depth perpendicular to the road (a ruler), mm:          p10 {report['perp_mm_p10_50_90'][0]:.1f}, median {report['perp_mm_p10_50_90'][1]:.1f}, p90 {report['perp_mm_p10_50_90'][2]:.1f}")
            print(f"   camera tilt from straight down: median {report['tilt_deg_p10_50_90'][1]:.0f} deg; axis depth / ruler depth: median {ratio.median():.2f}")
        print(f"   camera to pothole: median {report['camera_mm_p10_50_90'][1]:.0f} mm (p10 {report['camera_mm_p10_50_90'][0]:.0f}, p90 {report['camera_mm_p10_50_90'][2]:.0f})")
        print(f"   road ring around a pothole: rms {d.ring_rms_mm.median():.2f} mm (median)")
        if not args.inspect_only:
            d.to_csv(os.path.join(OUT_DIR, f"{args.source}_potholes.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"{args.source}_inspection.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote ml_results/depth_sources/{args.source}_inspection.json"
          + ("" if args.inspect_only else f", {args.source}_potholes.csv, and {len(os.listdir(cache))} cached frames"))


if __name__ == "__main__":
    main()
