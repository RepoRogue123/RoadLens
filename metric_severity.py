"""
Serving side of the measured-label depth models.

Two estimators of a pothole's bowl depth in millimetres, both trained against
PothRGBD RealSense measurements — the only severity signals in RoadLens trained
against measurement (the five legacy voters learned Depth-Anything pseudo-labels):

  relief      relief_depth.py — Depth-Anything-V2 fine-tuned on the dense RealSense
              maps. PREFERRED: on held-out capture sessions, 8.70 mm error against
              the regressor's 11.25 mm, and 22 of 29 Deep potholes found against 8.
  regressor   MoGe geometry -> hand-built metric features -> gradient boosting
              (scripts/train_metric_regressor.py). The fallback when the relief
              weights are missing, and still the source of the out-of-distribution
              check, camera distance and diameter.

The depth is turned into a severity with cut-offs chosen for the project's safety
policy (under-reporting costs three times over-reporting), each estimator with its own.

Gate pattern: `HAS_METRIC` is False if neither estimator can run, and every call
returns None rather than raising, so the API degrades to the legacy vote and says so.
Both models run once per image (`geometry_for_image`), at the 640-pixel scale they were
trained at; `predict` is then cheap per pothole.
"""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

import relief_depth
from metric_features import (FEATURE_COLS, SHAPE_COLS, extract_metric_features,
                             shape_features, to_train_scale)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "ml_models", "metric", "depth_regressor.joblib")
META_PATH = os.path.join(PROJECT_DIR, "ml_models", "metric", "depth_regressor_meta.json")

# Label bands (what "Shallow / Moderate / Deep" MEAN, in measured millimetres).
SHALLOW_MAX_MM = 25.0
MODERATE_MAX_MM = 50.0
MAX_PLAUSIBLE_MM = 250.0

# How the served verdict is chosen. "metric" (default when a model is present)
# makes the measured-label depth the headline severity; "legacy" restores the
# rule + pseudo-label vote. Evidence: MASTER_DOC changelog 2026-09-24 / 25 and
# 2026-10-02, ml_results/pothrgbd/served_severity_comparison.json, relief_eval_ens23.json.
SEVERITY_MODE = os.getenv("ROADLENS_SEVERITY_MODE", "metric").strip().lower()
# ROADLENS_DEPTH_SOURCE=regressor forces the MoGe regressor even when relief weights exist.
DEPTH_SOURCE = os.getenv("ROADLENS_DEPTH_SOURCE", "relief").strip().lower()
# A pothole covering under 5% of the photo is read from a close-up cut around it
# (relief_depth.bowl_depth_zoomed). On the Fan stereo frames, a second camera, this lowered
# the error from 15.3 to 12.4 mm; on PothRGBD only 18 test potholes are that small and it
# changed their error by -0.3 mm. MASTER_DOC changelog 2026-10-03. ROADLENS_ZOOM_SMALL=0 turns it off.
ZOOM_SMALL = os.getenv("ROADLENS_ZOOM_SMALL", "1").strip() != "0"

try:
    import moge_backend
    _HAS_MOGE = bool(moge_backend.HAS_MOGE)
except Exception:  # pragma: no cover - import guard
    moge_backend = None
    _HAS_MOGE = False

HAS_REGRESSOR = _HAS_MOGE and os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
HAS_RELIEF = relief_depth.HAS_RELIEF and DEPTH_SOURCE != "regressor"
HAS_METRIC = HAS_REGRESSOR or HAS_RELIEF

_MODEL = None
_META: Optional[Dict[str, Any]] = None


def _load():
    """The MoGe-feature regressor and its metadata (None, None if unavailable)."""
    global _MODEL, _META
    if _MODEL is None and HAS_REGRESSOR:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _META = json.load(f)
    return _MODEL, _META


def severity_from_mm(mm: float) -> str:
    """The label definition: measured bowl depth to severity."""
    if mm < SHALLOW_MAX_MM:
        return "Shallow"
    if mm < MODERATE_MAX_MM:
        return "Moderate"
    return "Deep"


def decide(pred_mm: float, meta: Dict[str, Any]) -> str:
    """
    The served decision on a PREDICTED depth.

    Not the label bands: every estimator pulls predictions toward the middle, and
    banding the regressor at 25 / 50 mm found 12 of 114 measured-Deep potholes. The
    cut-offs come from the estimator's own metadata, chosen on held-out predictions to
    minimise under-reporting at three times the cost of over-reporting.
    """
    t = meta.get("decision_thresholds_mm") or {}
    t_mod = float(t.get("moderate_from", SHALLOW_MAX_MM))
    t_deep = float(t.get("deep_from", MODERATE_MAX_MM))
    if pred_mm < t_mod:
        return "Shallow"
    if pred_mm < t_deep:
        return "Moderate"
    return "Deep"


def is_primary() -> bool:
    """True when the metric verdict should be the headline severity."""
    return HAS_METRIC and SEVERITY_MODE == "metric"


# How far past its training range a single feature may sit, as a fraction of
# the range's width, before the photo counts as out of distribution.
FAR_OUT_FRACTION = 0.10

# Camera-to-pothole distance in the measured photos, from the RealSense readings
# themselves (5th to 95th percentile). This is what the user is told. MoGe's own distance
# estimate is NOT in real metres: on PothRGBD it reads a median 2.6 times too far
# (1.7-3.4 m for photos taken at 0.4-1.6 m) and tracks the true distance only weakly
# (r = 0.34). It is still a valid like-for-like test of "does this photo resemble the
# training photos", because both sides of that comparison are MoGe's, but the number
# itself must not be shown as a distance.
TRAINED_CAMERA_RANGE_MM = (400, 1650)        # 404 and 1,645 mm over the 1,051 measured potholes


def _out_of_range(feats: Dict[str, float], ranges: Dict[str, List[float]]):
    """Names of features outside the training range, and whether any is far outside."""
    outside, far = [], False
    for c, (lo, hi) in ranges.items():
        width = hi - lo
        if width <= 0:          # constant in training (e.g. valid fraction); no range to judge
            continue
        excess = max(lo - feats[c], feats[c] - hi) / width
        if excess > 0:
            outside.append(c)
            far = far or excess > FAR_OUT_FRACTION
    return outside, far


def geometry_for_image(image_bgr: np.ndarray, masks: List[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Run the relief network and MoGe once, at training scale. Masks come back at that scale."""
    if not HAS_METRIC:
        return None
    try:
        img, scaled = to_train_scale(image_bgr, masks)
        # the original photo is kept for close-up readings of small potholes: full detail
        geo: Dict[str, Any] = {"masks": scaled, "relief": None, "depth": None, "photo": image_bgr}
        if HAS_RELIEF:
            geo["relief"] = relief_depth.predict_relief(img)
        if HAS_REGRESSOR:
            g = moge_backend.get_geometry(img)
            if g is not None:
                geo.update(depth=g["depth"], intrinsics=g["intrinsics"],
                           normal=g.get("normal"), valid=g.get("mask"))
        return geo if (geo["relief"] is not None or geo["depth"] is not None) else None
    except Exception as e:
        print(f"metric_severity: geometry failed: {e}")
        return None


def predict(geometry: Optional[Dict[str, Any]], index: int) -> Optional[Dict[str, Any]]:
    """Predicted bowl depth, an empirical error bar and a severity for mask `index`."""
    if geometry is None:
        return None
    try:
        mask = geometry["masks"][index]
        out: Dict[str, Any] = {}

        # MoGe side: the regressor's own estimate, plus what only MoGe can tell us —
        # whether this photo resembles the measured training photos at all.
        reg_mm = None
        model, meta = _load()
        if model is not None and geometry["depth"] is not None:
            mf = extract_metric_features(mask, geometry["depth"], geometry["intrinsics"],
                                         normal=geometry["normal"], valid=geometry["valid"])
            sf = shape_features(mask)
            if mf is not None and sf is not None:
                feats = {**mf, **sf}
                x = np.array([[feats[c] for c in meta["features"]]], dtype=np.float64)
                reg_mm = float(np.clip(model.predict(x)[0], 0.0, MAX_PLAUSIBLE_MM))
                outside, far = _out_of_range(feats, meta.get("feature_ranges") or {})
                out.update({
                    "rawPlaneP90Mm": round(mf["mf_p90_mm"], 1),
                    "cameraDistanceMm": round(mf["mf_cam_dist_mm"], 0),
                    "diameterMm": round(mf["mf_diam_mm"], 0),
                    # Features outside the training range (1st-99th percentile). One far
                    # outside, or several slightly outside, means the photo is unlike
                    # PothRGBD and the estimate is an extrapolation that the error bar
                    # does not cover. This applies to the relief network just as much:
                    # it was trained on the same photos.
                    "outOfRange": outside,
                    "inDistribution": not far and len(outside) < 2,
                    "trainedCameraRangeMm": list(TRAINED_CAMERA_RANGE_MM),
                    "cameraFurtherThanTraining": bool(
                        mf["mf_cam_dist_mm"] > ((meta.get("feature_ranges") or {}).get("mf_cam_dist_mm") or [0, np.inf])[1]),
                    "regressorDepthMm": round(reg_mm, 1),
                })

        relief_mm, read_from = None, None
        if geometry["relief"] is not None:
            b = relief_depth.bowl_depth_mm(geometry["relief"], mask)
            read_from = "frame"
            if ZOOM_SMALL and geometry.get("photo") is not None:
                z = relief_depth.bowl_depth_zoomed(geometry["photo"], mask)
                if z is not None:
                    b, read_from = z, "close-up"
            if b is not None:
                relief_mm = float(np.clip(b, 0.0, MAX_PLAUSIBLE_MM))

        if relief_mm is not None:
            mm, source, used = relief_mm, "relief", relief_depth.meta()
        elif reg_mm is not None:
            mm, source, used = reg_mm, "regressor", meta
        else:
            return None
        q80 = float(used.get("abs_error_q80_mm", np.nan))
        out.update({
            "depthMm": round(mm, 1),
            "intervalMm": round(q80, 1),
            "lowMm": round(max(mm - q80, 0.0), 1),
            "highMm": round(mm + q80, 1),
            "severity": decide(mm, used),
            "bandOfEstimate": severity_from_mm(mm),
            "source": source,
            "readFrom": read_from if source == "relief" else None,
        })
        # Without MoGe there is nothing to judge the photo against: unknown, not "fine".
        out.setdefault("inDistribution", None)
        out.setdefault("outOfRange", [])
        return out
    except Exception as e:
        print(f"metric_severity: prediction failed: {e}")
        return None


def describe() -> str:
    if not HAS_METRIC:
        return "measured-label depth model OFF (needs relief weights, or MoGe plus ml_models/metric/)"
    parts = []
    if HAS_RELIEF:
        parts.append(relief_depth.describe())
    if HAS_REGRESSOR:
        _m, meta = _load()
        yolo = meta.get("cv", {}).get("yolo", {}).get("mae", {})
        parts.append(("fallback / photo check: " if HAS_RELIEF else "")
                     + f"MoGe regressor ({meta.get('config')}, CV MAE {yolo.get('mean', float('nan')):.1f} mm)")
    else:
        parts.append("MoGe unavailable: no out-of-distribution check")
    return "; ".join(parts) + f"; mode={SEVERITY_MODE}"


__all__ = ["HAS_METRIC", "HAS_RELIEF", "HAS_REGRESSOR", "SEVERITY_MODE", "geometry_for_image",
           "predict", "describe", "decide", "is_primary", "severity_from_mm", "FEATURE_COLS", "SHAPE_COLS"]
