"""
Measured error distributions for grading ANY photo, written to one small file that ships.

The CVPR severity framework (scripts/cvpr_severity_framework.py) grades every photo, not
only photos like the measured training set, and reports how sure it is as probabilities.
A probability needs to know how wrong each measurement can be, and how wrong depends on
whether the photo resembles the training photos. Every distribution here is empirical,
from held-out data with measured truth:

    depth, in range       served depth minus RealSense depth, PothRGBD test sessions
                          (scripts/eval_zoom_inference.py --tag served: the installed
                          networks with the close-up reading, as the app serves them)
    depth, other camera   the same on the Fan stereo frames, a camera the network never saw:
                          the error used for photos unlike the training set
    angular size          MoGe angular size / true angular size, Fan point clouds
                          (scripts/eval_diameter_fan.py), used for the width when the camera
                          distance is known

Also recorded, for the app to quote: the wheel-drop accuracy against measured 3D on the
PothRGBD test sessions (scripts/eval_wheel_drop.py), per tyre.

Camera distance is NOT modelled for unknown photos: on three cameras with measured distance
no single-photo model was reliable (scripts/eval_camera_distance.py; per-camera median ratio
predicted / true: MoGe-3 5.6 / 2.6 / 1.4, Depth-Anything-V2 metric 10.0 / 4.9 / 1.9, Depth Pro
2.9 / 1.2 / 0.47 on Fan / PothRGBD / RSRD). Those results are recorded here for reference.

Each distribution is stored as 101 quantiles (inverse-CDF sampling at run time) with its n
and source. The file holds no data and is committed beside relief_meta.json.

Output: ml_models/metric/severity_error_model.json

Usage:
    python scripts/build_severity_error_model.py
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_DIR, "ml_results", "depth_sources")
OUT = os.path.join(PROJECT_DIR, "ml_models", "metric", "severity_error_model.json")
Q = np.linspace(0, 100, 101)


def quantiles(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {"n": int(v.size), "q": [round(float(x), 4) for x in np.percentile(v, Q)]}


def distance_findings():
    out = {}
    for tag in ("local", "depthpro"):
        p = os.path.join(SRC, f"camera_distance_{tag}.json")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                for model, by_cam in json.load(fh).items():
                    out[model] = {cam: {"n": s["n"], "median_ratio": round(s["median_ratio"], 2),
                                        "within25": round(s["within25"], 2)} for cam, s in by_cam.items()}
    return out


def wheel_drop_validation():
    p = os.path.join(SRC, "wheel_drop.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        src = json.load(fh)["sources"].get("pothrgbd")
    if not src:
        return None
    return {"source": "PothRGBD held-out test sessions, RealSense relief (scripts/eval_wheel_drop.py)",
            "n": src["n_potholes"],
            "tyres": {t: {"mae_mm": round(v["mae"], 1), "bias_mm": round(v["bias"], 1), "r": round(v["r"], 2)}
                      for t, v in src["tyres"].items()}}


def main():
    z = pd.read_csv(os.path.join(SRC, "zoom_eval_served.csv"))
    in_range = z[z.set == "pothrgbd test"]
    other = z[z.set == "fan"]
    f = pd.read_csv(os.path.join(SRC, "diameter_fan.csv"))
    angular = (f.moge_diam_mm / f.moge_dist_mm) / (f.true_diam_mm / f.true_dist_mm)

    model = {
        "fitted": str(date.today()),
        "depth_error_mm": {
            "in_range": {**quantiles(in_range.served - in_range.gt_mm),
                         "source": "PothRGBD test sessions, served networks with close-up reading (predicted - measured)"},
            "other_camera": {**quantiles(other.served - other.gt_mm),
                             "source": "Fan stereo frames, a camera the network never saw (predicted - measured)"},
        },
        "angular_size_ratio": {**quantiles(angular), "source": "MoGe / true angular size, Fan point clouds"},
        "camera_distance_findings": distance_findings(),
        "wheel_drop_validation": wheel_drop_validation(),
        "notes": [
            "Depth errors are additive in mm; the angular-size error is a ratio.",
            "'other_camera' rests on one stereo rig (79 potholes): an estimate of the error on unfamiliar photos, not a guarantee.",
            "No single-photo model estimated camera distance reliably across cameras, so width needs a supplied distance.",
        ],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    for name, d in model["depth_error_mm"].items():
        q = d["q"]
        print(f"depth error {name:12s}: n={d['n']:3d}  p10 {q[10]:+6.1f}  median {q[50]:+6.1f}  p90 {q[90]:+6.1f} mm")
    a = model["angular_size_ratio"]
    print(f"angular size ratio: n={a['n']}  p10 {a['q'][10]:.2f}  median {a['q'][50]:.2f}  p90 {a['q'][90]:.2f}")
    print(f"wrote {os.path.relpath(OUT, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
