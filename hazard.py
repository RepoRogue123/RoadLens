"""
Pothole hazard from physics and from a published risk policy, with nothing tuned by hand.

Two parts, both shared by the API and scripts/cvpr_severity_framework.py:

1. WHEEL DROP (Method 3A). How far a wheel's centre falls when it rolls across the
   pothole. A wheel cannot follow a hole narrower than itself all the way down: it bridges
   it and dips only as far as its circle fits. The drop is computed by rolling a circle of
   the tyre's radius over the pothole's measured or predicted cross-section; across its
   width the tyre rests on the highest point under it, so a narrow tyre falls further into a
   narrow hole than a wide one. Tyre sizes come from standard size codes
   (diameter = rim x 25.4 + 2 x section width x aspect ratio), not from tuning.

   The drop needs the cross-section in millimetres along the road, which needs the photo's
   scale: millimetres per pixel = (angular size x camera distance) / pothole width in pixels.
   With no camera distance there is no horizontal scale, and the drop is reported only as
   its upper bound (a wheel can never drop further than the pothole is deep).

2. RISK MATRIX (Method 2A). Response category from the risk assessment of the UK code of
   practice "Well-managed Highway Infrastructure" (2016), as published by Essex Highways,
   Maintenance & Inspections Strategy 2026, section 1.5-1.6: risk = likelihood (1-5) x
   consequence (1-5). Observations 1-9 (next routine inspection), D3 10-16 (90 working
   days), D2 20 (2 working days on county routes, 5 on local routes), D1 25 (2 hours).
   Investigatory level for a carriageway pothole: at least 100 mm across in two directions
   and at least 50 mm deep (cycle lanes: 75 mm and 20 mm).
   The code leaves likelihood and consequence to the inspector. Here likelihood is an
   input (where the pothole is), and consequence is derived from measurements by a
   provisional rule stated in `consequence_from_measurements`, to be replaced by the weights
   learned from the team's labels (scripts/fit_priority_model.py).
"""
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

# ── Tyres: (label, size code, rim inches, section width mm, aspect %) ──────────
# Common sizes for each kind of vehicle. Bias-ply moped and rickshaw tyres have a nominal
# 100% aspect ratio; 700x25C is ISO 25-622 (622 mm bead seat diameter).
_TYRE_SPECS = [
    ("bicycle", "Bicycle", "700x25C (ISO 25-622)", None, 25, None),
    ("scooter", "Scooter", "90/90-10", 10, 90, 90),
    ("auto_rickshaw", "Auto-rickshaw", "4.00-8", 8, 101.6, 100),
    ("motorcycle", "Motorcycle (front)", "120/70-17", 17, 120, 70),
    ("car", "Car", "195/65 R15", 15, 195, 65),
    ("suv", "SUV", "235/65 R17", 17, 235, 65),
    ("truck", "Truck / bus", "295/80 R22.5", 22.5, 295, 80),
]


def _diameter_mm(rim_in, width_mm, aspect):
    if rim_in is None:                     # bicycle: ISO bead seat diameter + 2 x width
        return 622.0 + 2.0 * width_mm
    return rim_in * 25.4 + 2.0 * width_mm * aspect / 100.0


TYRES: List[Dict] = [
    {"id": tid, "label": label, "size": size, "width_mm": float(w),
     "radius_mm": round(_diameter_mm(rim, w, asp) / 2.0, 1)}
    for tid, label, size, rim, w, asp in _TYRE_SPECS
]


def wheel_centre(height_mm: np.ndarray, step_mm: float, radius_mm: float) -> np.ndarray:
    """
    Height of a wheel's centre at each point of a 1-D surface profile.

    `height_mm` is the surface height along the travel line (0 = road, negative = below the
    road), sampled every `step_mm`. At each position the wheel centre sits as high as the
    surface under its circle requires: max over nearby points of height + sqrt(R^2 - dx^2).
    On flat road that is R.
    """
    h = np.asarray(height_mm, dtype=np.float64)
    half = int(np.ceil(radius_mm / step_mm))
    offs = np.arange(-half, half + 1)
    lift = np.sqrt(np.clip(radius_mm ** 2 - (offs * step_mm) ** 2, 0.0, None))
    pad = np.concatenate([np.zeros(half), h, np.zeros(half)])      # flat road beyond the profile
    windows = np.lib.stride_tricks.sliding_window_view(pad, offs.size)
    return (windows + lift[None, :]).max(axis=1)


def wheel_drop_profile(height_mm: np.ndarray, step_mm: float, radius_mm: float) -> float:
    """Largest fall of a wheel centre rolling along a 1-D profile: R minus its lowest centre height."""
    if np.asarray(height_mm).size == 0:
        return 0.0
    return float(max(0.0, radius_mm - wheel_centre(height_mm, step_mm, radius_mm).min()))


def band_profile(relief_mm: np.ndarray, mask: np.ndarray, axis: int, tyre_width_px: int) -> np.ndarray:
    """
    Surface height along a travel line through the pothole, for a tyre of given width.

    `relief_mm` is depth below the road (positive = deeper). The travel line runs along
    `axis` (0 = image rows, i.e. the road runs up the image; 1 = columns) through the
    pothole's deepest point. Across the tyre's width the tyre rests on the HIGHEST surface
    under it, so the profile is the minimum relief over that band. Outside the pothole
    the road is flat (relief 0).
    """
    r = np.where(mask > 0, np.asarray(relief_mm, dtype=np.float64), 0.0)
    r = np.clip(r, 0.0, None)                         # bumps above the road do not lower a wheel
    yy, xx = np.unravel_index(np.argmax(r), r.shape)
    half = max(0, tyre_width_px // 2)
    if axis == 0:
        band = r[:, max(0, xx - half):xx + half + 1]
        line = band.min(axis=1)
    else:
        band = r[max(0, yy - half):yy + half + 1, :]
        line = band.min(axis=0)
    return -line                                      # surface height: negative below the road


def wheel_drops(relief_mm: Optional[np.ndarray], mask: np.ndarray, depth_mm: float,
                mm_per_px: Optional[float], axes=(0, 1)) -> Dict:
    """
    Drop for every tyre in TYRES. With a scale (mm per pixel), the wheel is rolled over the
    pothole's cross-section along each candidate travel direction and the worst is kept;
    without one, only the upper bound min(deepest point, radius) is known. The bound uses the
    deepest point, not the reported depth (90th percentile): a narrow tyre in a wide hole
    reaches the bottom.
    """
    out = {"scaled": mm_per_px is not None and relief_mm is not None, "tyres": []}
    deepest = depth_mm
    if relief_mm is not None and (mask > 0).any():
        deepest = max(depth_mm, float(np.max(np.asarray(relief_mm)[mask > 0])))
    out["deepest_mm"] = round(deepest, 1)
    for t in TYRES:
        bound = min(deepest, t["radius_mm"])
        rec = {**t, "drop_bound_mm": round(bound, 1), "drop_mm": None, "profile_axis": None}
        if out["scaled"]:
            width_px = max(1, int(round(t["width_mm"] / mm_per_px)))
            best = (0.0, None)
            for ax in axes:
                prof = band_profile(relief_mm, mask, ax, width_px)
                d = wheel_drop_profile(prof, mm_per_px, t["radius_mm"])
                if d >= best[0]:
                    best = (d, ax)
            rec["drop_mm"], rec["profile_axis"] = round(best[0], 1), best[1]
        out["tyres"].append(rec)
    return out


def relief_patch(relief_mm: np.ndarray, mask: np.ndarray, max_side: int = 96) -> Dict:
    """
    The pothole's relief around its outline, downsampled for the web page, which re-runs the
    same wheel roll in the browser for any width the user sets. Each patch pixel spans
    `px_per_cell` image pixels; values are mm below the road, 0 outside the outline.
    """
    ys, xs = np.nonzero(mask)
    pad = max(4, int(0.15 * max(np.ptp(ys), np.ptp(xs))))
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    r = np.where(mask > 0, np.clip(np.asarray(relief_mm, np.float64), 0, None), 0.0)[y0:y1, x0:x1]
    m = (mask[y0:y1, x0:x1] > 0).astype(np.uint8)
    s = min(1.0, max_side / max(r.shape))
    if s < 1.0:
        size = (max(1, int(round(r.shape[1] * s))), max(1, int(round(r.shape[0] * s))))
        r = cv2.resize(r.astype(np.float32), size, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, size, interpolation=cv2.INTER_NEAREST)
        r = np.where(m > 0, r, 0.0)
    return {"relief_mm": [[round(float(v), 1) for v in row] for row in r],
            "px_per_cell": round(1.0 / s, 4),
            "equiv_diameter_cells": round(float(2.0 * np.sqrt(m.sum() / np.pi)), 3)}


def pothole_hazard(relief_mm: Optional[np.ndarray], mask: np.ndarray, depth_mm: float,
                   angular_size: Optional[float], camera_distance_mm: Optional[float]) -> Dict:
    """
    Wheel drop for every tyre, for one pothole. The scale is the pothole's width over its
    width in pixels, the width being angular size x camera distance; without a distance only
    the upper bound is given. `relief_mm` and `mask` must be at the same resolution.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    eq_px = 2.0 * np.sqrt(m.sum() / np.pi)
    width_mm = angular_size * camera_distance_mm if angular_size and camera_distance_mm else None
    mm_per_px = width_mm / eq_px if width_mm and eq_px > 0 else None
    out = wheel_drops(relief_mm, m, depth_mm, mm_per_px)
    out.update(width_mm=None if width_mm is None else round(width_mm, 0),
               mm_per_px=None if mm_per_px is None else round(mm_per_px, 4),
               angular_size=None if angular_size is None else round(angular_size, 5),
               camera_distance_mm=camera_distance_mm, depth_mm=round(depth_mm, 1))
    if relief_mm is not None and m.any():
        out["patch"] = relief_patch(relief_mm, m)
    return out


# ── Risk matrix (UK code of practice, Essex Highways 2026) ─────────────────────
LIKELIHOOD = {1: "Rare", 2: "Unlikely", 3: "Possible", 4: "Likely", 5: "Almost certain"}
CONSEQUENCE = {1: "Insignificant", 2: "Minor", 3: "Moderate", 4: "Significant", 5: "Major"}
ROUTES = {"county": "County route", "local": "Local route"}
INVESTIGATORY = {"carriageway": {"across_mm": 100, "depth_mm": 50}, "cycleway": {"across_mm": 75, "depth_mm": 20}}


def response_for(score: int, route: str = "local") -> Dict:
    """Category and response time for a risk score (likelihood x consequence)."""
    if score >= 25:
        return {"category": "D1", "response": "2 hours", "order": 3}
    if score >= 20:
        return {"category": "D2", "response": "2 working days" if route == "county" else "5 working days", "order": 2}
    if score >= 10:
        return {"category": "D3" + ("A" if score in (15, 16) else ""), "response": "90 working days", "order": 1}
    return {"category": "Observation", "response": "next routine inspection", "order": 0}


def consequence_from_measurements(p_investigatory: float, car_drop_mm: Optional[float]) -> Dict:
    """
    PROVISIONAL consequence score (1-5) from measurements, until learned weights replace it.

    Anchored on the code's own investigatory level: the probability that the pothole reaches
    50 mm (from the measured depth error) sets the score, and a car-wheel drop of 40 mm or
    more, when the scale is known, raises it by one (a drop of that size is what the 50 mm
    level is meant to catch for a tyre that bridges part of the hole).
    """
    c = 1 + int(round(4 * float(np.clip(p_investigatory, 0, 1))))
    if car_drop_mm is not None and car_drop_mm >= 40.0:
        c = min(5, c + 1)
    return {"score": c, "label": CONSEQUENCE[c], "rule": "provisional: investigatory-level probability (+1 if car wheel drops >= 40 mm)"}


# ── Learned replacements (Methods 2B and 3B), fitted on the team's labels ──────
# The same seven measurements feed both models. They are computed for every photo by the
# pipeline, stored with each labelled pothole (data/hazard_pool.csv), and read back here.
#   depth_mm        served bowl depth (90th percentile below the road)
#   deepest_mm      deepest point of the predicted relief inside the outline
#   water_coverage  share of the pothole the water segmenter covers
#   edge_sharpness  rim sharpness (temporal_analysis), a sign of recent failure
#   crack_texture   cracking around the rim (temporal_analysis)
#   dinov2_ratio    DINOv2 texture ratio, inside vs ring (semantic_config.RATIO_KEY)
#   log_area_frac   log of the share of the photo the pothole covers
FEATURES = ["depth_mm", "deepest_mm", "water_coverage", "edge_sharpness", "crack_texture",
            "dinov2_ratio", "log_area_frac"]
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models", "hazard")
PRIORITY_MODEL = os.path.join(MODEL_DIR, "priority_model.json")      # scripts/fit_priority_model.py
PAIRS_MODEL = os.path.join(MODEL_DIR, "hazard_pairs_model.json")     # scripts/fit_hazard_pairs.py


def feature_row(depth_mm, deepest_mm, water_coverage, edge_sharpness, crack_texture, dinov2_ratio,
                area_frac) -> Dict[str, float]:
    """The model inputs for one pothole; None becomes NaN and is imputed by the model."""
    nan = float("nan")
    f = lambda v: nan if v is None else float(v)                         # noqa: E731
    return {"depth_mm": f(depth_mm), "deepest_mm": f(deepest_mm), "water_coverage": f(water_coverage),
            "edge_sharpness": f(edge_sharpness), "crack_texture": f(crack_texture),
            "dinov2_ratio": f(dinov2_ratio),
            "log_area_frac": float(np.log(max(float(area_frac), 1e-6))) if area_frac is not None else nan}


def _standardise(model: Dict, row: Dict[str, float]) -> np.ndarray:
    x = np.array([row[k] for k in model["features"]], dtype=np.float64)
    med = np.array(model["impute_median"], dtype=np.float64)
    x = np.where(np.isfinite(x), x, med)
    return (x - np.array(model["mean"])) / np.array(model["scale"])


_LOADED: Dict[str, Optional[Dict]] = {}


def _load(path: str) -> Optional[Dict]:
    if path not in _LOADED:
        _LOADED[path] = None
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                m = json.load(fh)
            _LOADED[path] = m if m.get("adopted") else None                  # only a model that passed its rule
    return _LOADED[path]


def learned_consequence(row: Dict[str, float]) -> Optional[Dict]:
    """
    Method 2B: consequence 1-5 from the ordinal model fitted on the team's ratings, or None
    when no adopted model exists. Ordinal: one logistic model per threshold, P(c > k).
    """
    m = _load(PRIORITY_MODEL)
    if m is None:
        return None
    z = _standardise(m, row)
    above = [1.0 / (1.0 + np.exp(-(float(np.dot(w, z)) + b))) for w, b in zip(m["weights"], m["bias"])]
    above = np.minimum.accumulate(np.clip(above, 0, 1))                    # P(c > k) cannot rise with k
    expected = 1.0 + float(np.sum(above))
    c = int(np.clip(round(expected), 1, 5))
    return {"score": c, "label": CONSEQUENCE[c], "expected": round(expected, 2),
            "rule": f"learned from {m['n_labels']} team ratings (cross-fitted weighted kappa {m['cv_kappa']:.2f})"}


def learned_hazard(row: Dict[str, float]) -> Optional[Dict]:
    """
    Method 3B: Bradley-Terry hazard score from the team's pairwise judgements, or None when
    no adopted model exists. Reported as the percentile among the labelled potholes.
    """
    m = _load(PAIRS_MODEL)
    if m is None:
        return None
    s = float(np.dot(m["weights"], _standardise(m, row)))
    pct = float(np.searchsorted(np.sort(m["pool_scores"]), s) / max(1, len(m["pool_scores"])))
    return {"score": round(s, 3), "percentile": round(100 * pct, 0),
            "rule": f"learned from {m['n_pairs']} team comparisons (held-out agreement {m['cv_agreement']:.0%})"}


def risk_assessment(likelihood: int, consequence: int, route: str = "local") -> Dict:
    score = int(likelihood) * int(consequence)
    return {"likelihood": int(likelihood), "likelihood_label": LIKELIHOOD[int(likelihood)],
            "consequence": int(consequence), "consequence_label": CONSEQUENCE[int(consequence)],
            "score": score, "route": route, **response_for(score, route),
            "source": "Well-managed Highway Infrastructure (2016) risk matrix, as published by Essex Highways 2026"}
