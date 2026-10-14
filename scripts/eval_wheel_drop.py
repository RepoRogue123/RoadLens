"""
How accurate is the predicted wheel drop? Measured against measured 3D.

hazard.wheel_drops rolls each tyre over the pothole's cross-section. For every tyre in
hazard.TYRES the drop from the MEASURED cross-section is compared with the drop from the
PREDICTED one (the served relief networks), at the same scale (millimetres per pixel):

    fan        stereo point clouds: true relief (exact road plane) and true scale (spacing of
               neighbouring 3D points). Also tests the scale the framework uses when the camera
               distance is supplied: MoGe angular size x true distance, against the true scale.
    pothrgbd   held-out test sessions (the networks never saw them): RealSense relief, and
               scale = RealSense distance / MoGe focal length in pixels (MoGe's angular size is
               within a median 2% on Fan, scripts/eval_diameter_fan.py). Its potholes are larger
               than Fan's, so tyres actually drop into them.

Reported per source and tyre: the median true drop, mean absolute error and bias of the
predicted drop, and its error when the scale is off by the 10th / 90th percentile of the
angular-size ratio (the error the framework carries when a distance is supplied). Also the
share of true drops covered by the no-scale upper bound min(predicted depth, wheel radius).

Caveats: Fan is one stereo rig with small potholes (median 81 mm across); PothRGBD's scale
rests on MoGe's focal length; travel direction is unknown, so both image axes are tried and
the worse one kept, for truth and prediction alike.

Output: ml_results/depth_sources/wheel_drop.csv and wheel_drop.json

Usage:
    python scripts/eval_wheel_drop.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import hazard                                                                    # noqa: E402
import moge_backend                                                              # noqa: E402
import relief_depth as R                                                         # noqa: E402
from metric_features import MIN_INSIDE_PX                                        # noqa: E402
from scripts.build_relief_targets import OUT_DIR, to_scale                       # noqa: E402
from scripts.build_severity_error_model import OUT as ERROR_MODEL                # noqa: E402
from scripts.depth_sources import fan_frames                                     # noqa: E402
from scripts.eval_camera_distance import pothrgbd_items, ring_median             # noqa: E402
from scripts.train_relief_depth import relief_target                             # noqa: E402


def fan_scale(xyz, mask):
    """Median spacing between horizontally adjacent measured points around the pothole."""
    ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
    a, b = xyz[:, :-1], xyz[:, 1:]
    ok = np.isfinite(a).all(-1) & np.isfinite(b).all(-1) & ring[:, :-1]
    if ok.sum() < 50:
        return None
    return float(np.median(np.linalg.norm((a - b)[ok], axis=-1)))


def clean_relief(relief, valid):
    """Measured relief with sensor holes filled from their surroundings and speckle removed."""
    r = np.where(valid, relief, 0).astype(np.float32)
    holes = (~valid).astype(np.uint8)
    if holes.any():
        r = cv2.inpaint(r, holes, 5, cv2.INPAINT_NS)
    return cv2.medianBlur(r, 5)


def frames():
    """(source, key, bgr, true relief, polys, scale_fn(mask) -> true mm/px, angular-scale_fn or None)."""
    ang = pd.read_csv(os.path.join(OUT_DIR, "diameter_fan.csv"))
    ang = {(r.key, int(r.pothole)): (r.moge_diam_mm / r.moge_dist_mm, r.true_dist_mm) for r in ang.itertuples()}
    for raw in fan_frames():
        f = to_scale(raw)
        n, comp = cv2.connectedComponents(f["mask"])
        polys = [(comp == c).astype(np.uint8) for c in range(1, n)]
        t = relief_target(f["depth_mm"].astype(np.float64), polys, exact=True)
        if t is None:
            continue

        def angular(m, c, key=f["key"]):
            if (key, c) not in ang:
                return None
            a, dist = ang[(key, c)]
            return a * dist / (2.0 * np.sqrt(m.sum() / np.pi))
        yield "fan", f["key"], f["bgr"], clean_relief(t[0], t[1]), polys, \
            (lambda m, c, xyz=f["xyz_mm"]: fan_scale(xyz, m)), angular
    for _src, key, bgr, depth, valid, polys in pothrgbd_items():
        t = relief_target(np.where(valid, depth, 0), polys, exact=True)
        g = moge_backend.get_geometry(bgr)
        if t is None or g is None:
            continue
        k = np.asarray(g["intrinsics"], np.float64)
        f_px = 0.5 * (k[0, 0] * depth.shape[1] + k[1, 1] * depth.shape[0])

        def scale(m, c, depth=depth, valid=valid, f_px=f_px):
            z = ring_median(depth, m, valid)
            return None if z is None else z / f_px
        yield "pothrgbd", key, bgr, clean_relief(t[0], t[1]), polys, scale, None


def main():
    with open(ERROR_MODEL, encoding="utf-8") as fh:
        q = json.load(fh)["angular_size_ratio"]["q"]
    lo, hi = q[10], q[90]
    nets = [R.build_model(w).to("cuda").eval() for w in R.installed_weights()]
    rows = []
    for src, key, bgr, true_relief, polys, scale_fn, angular_fn in frames():
        pred_relief = R.predict_relief(bgr, nets)
        for c, m in enumerate(polys, start=1):
            if int(m.sum()) < MIN_INSIDE_PX:
                continue
            s = scale_fn(m, c)
            bowl = R.bowl_depth_mm(pred_relief, m)
            if s is None or bowl is None:
                continue
            true_depth = float(np.percentile(true_relief[m > 0], 90))
            runs = {"true": hazard.wheel_drops(true_relief, m, true_depth, s),
                    "pred": hazard.wheel_drops(pred_relief, m, bowl, s),
                    "pred_scale_lo": hazard.wheel_drops(pred_relief, m, bowl, s * lo),
                    "pred_scale_hi": hazard.wheel_drops(pred_relief, m, bowl, s * hi)}
            s_ang = angular_fn(m, c) if angular_fn else None
            if s_ang is not None:
                runs["pred_angular"] = hazard.wheel_drops(pred_relief, m, bowl, s_ang)
            ys, xs = np.nonzero(m)
            across = float(max(np.ptp(ys), np.ptp(xs)) + 1) * s
            for i, t in enumerate(hazard.TYRES):
                row = {"source": src, "key": key, "pothole": c, "tyre": t["id"], "true_depth_mm": true_depth,
                       "pred_depth_mm": bowl, "across_mm": across, "bound": runs["pred"]["tyres"][i]["drop_bound_mm"]}
                row.update({name: r["tyres"][i]["drop_mm"] for name, r in runs.items()})
                rows.append(row)
        print(f"  {src} {key}: {len(rows) // len(hazard.TYRES)} potholes", flush=True)

    d = pd.DataFrame(rows)
    res = {"angular_ratio_p10_p90": [lo, hi], "sources": {}}
    for src, ds in d.groupby("source", sort=False):
        n = ds.groupby(["key", "pothole"]).ngroups
        res["sources"][src] = {"n_potholes": int(n), "across_mm_median": float(ds.across_mm.median()),
                               "true_depth_mm_median": float(ds.true_depth_mm.median()), "tyres": {}}
        print(f"\n{src}: {n} potholes, median {ds.across_mm.median():.0f} mm across, "
              f"{ds.true_depth_mm.median():.1f} mm deep. Wheel drop, mm:")
        print(f"{'tyre':14s} {'true med':>8s} {'true p90':>8s} | {'MAE':>5s} {'bias':>6s} {'r':>5s} | "
              f"{'MAE scale p10':>13s} {'p90':>5s} | {'bound covers':>12s}")
        for tid, g in ds.groupby("tyre", sort=False):
            e = g.pred - g.true
            r = {"true_median": float(g.true.median()), "true_p90": float(g.true.quantile(0.9)),
                 "mae": float(e.abs().mean()), "bias": float(e.mean()),
                 "r": float(np.corrcoef(g.pred, g.true)[0, 1]) if g.true.std() > 0 and g.pred.std() > 0 else None,
                 "mae_scale_lo": float((g.pred_scale_lo - g.true).abs().mean()),
                 "mae_scale_hi": float((g.pred_scale_hi - g.true).abs().mean()),
                 "bound_covers": float((g.bound >= g.true - 0.5).mean())}
            if "pred_angular" in g and g.pred_angular.notna().any():
                r["mae_angular_scale"] = float((g.pred_angular - g.true).abs().mean())
            res["sources"][src]["tyres"][tid] = r
            rs = f"{r['r']:5.2f}" if r["r"] is not None else "    -"
            print(f"{tid:14s} {r['true_median']:8.1f} {r['true_p90']:8.1f} | {r['mae']:5.1f} {r['bias']:+6.1f} {rs} | "
                  f"{r['mae_scale_lo']:13.1f} {r['mae_scale_hi']:5.1f} | {r['bound_covers']:12.0%}")
    d.to_csv(os.path.join(OUT_DIR, "wheel_drop.csv"), index=False)
    with open(os.path.join(OUT_DIR, "wheel_drop.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote ml_results/depth_sources/wheel_drop.json")


if __name__ == "__main__":
    main()
