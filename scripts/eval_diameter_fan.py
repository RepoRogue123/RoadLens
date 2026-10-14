"""
How far off is MoGe's pothole diameter? Measured against the Fan stereo point clouds.

The CVPR severity framework (scripts/cvpr_severity_framework.py) grades potholes by ASTM
D6433, which needs the pothole's DIAMETER in millimetres as well as its depth. The only
diameter available from one photo is MoGe's (metric_features.extract_metric_features,
`mf_diam_mm`), and MoGe's metric scale is known to drift: on PothRGBD it places the camera
a median 2.6 times too far (metric_severity.TRAINED_CAMERA_RANGE_MM). PothRGBD ships no
camera intrinsics, so its true diameters cannot be computed. The Fan stereo sets can: every
frame has a point cloud in millimetres and a pothole outline.

For each Fan pothole (connected component of the label, at the 640-px serving scale):

    true diameter   equivalent-circle diameter of the pothole's surface area, from the
                    measured point cloud (area of each pixel from its neighbour steps)
    MoGe diameter   the same quantity from MoGe's depth and intrinsics (mf_diam_mm),
                    exactly as the served features compute it

Both use the same definition, so the comparison is like for like. Reported: the median
ratio MoGe / true (the scale bias), its spread, the error before and after dividing by the
median ratio, and the same for camera distance.

Caveat: 79 potholes, one stereo rig, close range, looking down. A correction learned here
is checked by leave-one-dataset-out (Fan has three sets) but is not evidence for other
cameras.

Output: ml_results/depth_sources/diameter_fan.csv and diameter_fan.json

Usage:
    python scripts/eval_diameter_fan.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from metric_features import MIN_INSIDE_PX, _pixel_areas, extract_metric_features   # noqa: E402
from scripts.build_relief_targets import OUT_DIR, to_scale                         # noqa: E402
from scripts.depth_sources import PRESENT, fan_frames                              # noqa: E402


def true_geometry(xyz_mm, mask):
    """(equivalent diameter mm, median camera distance mm) of one pothole from the point cloud."""
    pts = np.nan_to_num(xyz_mm.astype(np.float64), nan=0.0)
    ok = np.isfinite(xyz_mm).all(-1) & (xyz_mm[..., 2] > 0)
    areas = _pixel_areas(pts)
    # a pixel's area is only trusted when it and its right and lower neighbours were measured
    good = ok & np.roll(ok, -1, 1) & np.roll(ok, -1, 0)
    inside = (mask > 0) & good
    if int(inside.sum()) < MIN_INSIDE_PX:
        return None
    area_mm2 = float(areas[inside].sum()) * (int((mask > 0).sum()) / int(inside.sum()))
    return 2.0 * np.sqrt(area_mm2 / np.pi), float(np.median(xyz_mm[..., 2][inside]))


def main():
    if not PRESENT["fan"]:
        sys.exit("the Fan stereo sets are not on disk (see scripts/depth_sources.py)")
    import moge_backend
    if not moge_backend.HAS_MOGE:
        sys.exit("MoGe is unavailable; nothing to compare")

    rows = []
    for raw in fan_frames():
        f = to_scale(raw)
        g = moge_backend.get_geometry(f["bgr"])
        if g is None:
            continue
        depth = np.asarray(g["depth"], np.float64)
        if depth.shape != f["mask"].shape:
            depth = cv2.resize(depth, f["mask"].shape[::-1], interpolation=cv2.INTER_NEAREST)
        valid = np.asarray(g["mask"]) > 0.5
        if valid.shape != depth.shape:
            valid = cv2.resize(valid.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        n, comp = cv2.connectedComponents(f["mask"])
        for c in range(1, n):
            m = (comp == c).astype(np.uint8)
            truth = true_geometry(f["xyz_mm"], m)
            mf = extract_metric_features(m, depth, g["intrinsics"], valid=valid)
            if truth is None or mf is None:
                continue
            rows.append({"key": f["key"], "group": f["group"], "pothole": c, "flagged": f["flagged"],
                         "true_diam_mm": truth[0], "moge_diam_mm": mf["mf_diam_mm"],
                         "true_dist_mm": truth[1], "moge_dist_mm": mf["mf_cam_dist_mm"]})
        print(f"  {f['key']}: {len(rows)} potholes so far", flush=True)

    d = pd.DataFrame(rows)
    d["diam_ratio"] = d.moge_diam_mm / d.true_diam_mm
    d["dist_ratio"] = d.moge_dist_mm / d.true_dist_mm

    def summary(pred, truth):
        e = np.abs(pred - truth)
        return {"mae_mm": float(e.mean()), "median_abs_err_mm": float(np.median(e)),
                "median_rel_err": float(np.median(e / truth)), "r": float(np.corrcoef(pred, truth)[0, 1])}

    k = float(d.diam_ratio.median())
    # Leave-one-set-out: the correction for each Fan set is learned on the other two, so the
    # corrected error is not scored on the potholes that chose it.
    loo = np.empty(len(d))
    for grp in d.group.unique():
        out = (d.group == grp).values
        loo[out] = d.moge_diam_mm.values[out] / float(d.diam_ratio[~out].median())
    res = {
        "n": int(len(d)),
        "true_diam_mm_p10_50_90": [float(v) for v in np.percentile(d.true_diam_mm, [10, 50, 90])],
        "diam_ratio_median": k,
        "diam_ratio_p10_p90": [float(v) for v in np.percentile(d.diam_ratio, [10, 90])],
        "dist_ratio_median": float(d.dist_ratio.median()),
        "raw": summary(d.moge_diam_mm.values, d.true_diam_mm.values),
        "corrected_leave_one_set_out": summary(loo, d.true_diam_mm.values),
        "share_within_25pct_raw": float((np.abs(d.moge_diam_mm / d.true_diam_mm - 1) <= 0.25).mean()),
        "share_within_25pct_corrected": float((np.abs(loo / d.true_diam_mm - 1) <= 0.25).mean()),
    }
    print(f"\n{res['n']} Fan potholes; true diameter median {res['true_diam_mm_p10_50_90'][1]:.0f} mm "
          f"(p10 {res['true_diam_mm_p10_50_90'][0]:.0f}, p90 {res['true_diam_mm_p10_50_90'][2]:.0f})")
    print(f"MoGe / true diameter: median {k:.2f} (p10 {res['diam_ratio_p10_p90'][0]:.2f}, p90 {res['diam_ratio_p10_p90'][1]:.2f}); "
          f"camera distance ratio median {res['dist_ratio_median']:.2f}")
    for name in ("raw", "corrected_leave_one_set_out"):
        s = res[name]
        print(f"  {name:28s} MAE {s['mae_mm']:6.1f} mm  median rel err {s['median_rel_err']:.0%}  r {s['r']:.3f}")
    print(f"  within 25% of true: raw {res['share_within_25pct_raw']:.0%}, corrected {res['share_within_25pct_corrected']:.0%}")
    os.makedirs(OUT_DIR, exist_ok=True)
    d.assign(corrected_loo_mm=loo).to_csv(os.path.join(OUT_DIR, "diameter_fan.csv"), index=False)
    with open(os.path.join(OUT_DIR, "diameter_fan.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("wrote ml_results/depth_sources/diameter_fan.json")


if __name__ == "__main__":
    main()
