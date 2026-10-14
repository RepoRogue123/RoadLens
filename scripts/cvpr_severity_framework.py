"""
CVPR Research Extension: Multi-Modal Metric Defect Characterization Framework.
==============================================================================

This script implements the three proposed CVPR formulations in a standalone,
parallel test harness without modifying the production RoadLens codebase:

1. Method 1 (ASTM D6433 pothole severity, Table X1.1 and clause X1.17.1.2):
   Low / Medium / High from maximum depth against average diameter, as the standard
   defines them. Depth comes from the served depth network with its 80% error range;
   diameter from MoGe's angular size times a camera distance (validated on the Fan point
   clouds, scripts/eval_diameter_fan.py). Where the ranges span two cells the grade is
   reported as a range. No grade below 13 mm deep or 100 mm across (not a pothole in the
   table), and none for photos outside the depth network's training range.

2. Method 2 (response category, UK risk matrix):
   risk = likelihood x consequence (1-5 each) with the categories and response times of the
   UK code of practice "Well-managed Highway Infrastructure" (Essex Highways 2026).
   Likelihood is an input (where the pothole is); consequence is learned from the team's
   ratings once a model passes its adoption rule (2B), else a provisional rule anchored on
   the code's investigatory level (2A). Replaces the hand-set two-tier priority of PR #1.

3. Method 3 (wheel drop by tyre, physics):
   how far a bicycle, scooter, auto-rickshaw, motorcycle, car, SUV and truck wheel falls
   into the pothole, from rolling each tyre over the predicted cross-section (3A, validated
   on measured 3D: scripts/eval_wheel_drop.py), plus the team's pairwise hazard ranking
   once adopted (3B). Replaces the hand-weighted Road Hazard Score of PR #1.

Outputs:
   - High-resolution multi-metric diagnostic panel saved to:
     tests/cvpr/{stem}_cvpr_panel.png
   - Structured JSON manifest saved to:
     tests/cvpr/{stem}_cvpr_metrics.json

Usage:
   # 1. Run on default image configured in script:
   python scripts/cvpr_severity_framework.py

   # 2. Specify a custom image via CLI:
   python scripts/cvpr_severity_framework.py --image "path/to/pothole.jpg"

   # 3. Batch evaluate a folder:
   python scripts/cvpr_severity_framework.py --folder "path/to/folder" --limit 5
"""
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import features
import foundation_features
import hazard
import metric_features
import metric_severity
import relief_depth
import segmentation
import semantic_config
import temporal_analysis
import water_config
import water_detection
import water_segmenter

# ==============================================================================
# USER CONFIGURATION: CHANGE THIS PATH TO TEST ANY SPECIFIC IMAGE DIRECTLY
# ==============================================================================
DEFAULT_IMAGE_PATH = os.path.join(
    PROJECT_DIR,
    "datasets",
    "water_detection_test_dataset",
    "water_test_dataset",
    "IMG",
    "Pothole-001.jpg",
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "tests", "cvpr")
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 1: ASTM D6433 pothole severity (Table X1.1 and clause X1.17.1.2)
# ──────────────────────────────────────────────────────────────────────────────
# ASTM D6433-16, X1.17 Potholes. Severity of a pothole up to 750 mm across comes from
# Table X1.1, maximum depth against average diameter. Grades are Low / Medium / High.
ASTM_DEPTH_EDGES_MM = (13.0, 25.0, 50.0)          # rows: 13–25, >25–50, >50 mm
ASTM_DIAMETER_EDGES_MM = (100.0, 200.0, 450.0, 750.0)  # columns: 100–200, 200–450, 450–750 mm
ASTM_TABLE_X1_1 = (
    ("L", "L", "M"),    # 13 to <= 25 mm
    ("L", "M", "H"),    # > 25 to <= 50 mm
    ("M", "M", "H"),    # > 50 mm
)
ASTM_GRADE_NAMES = {"L": "Low", "M": "Medium", "H": "High"}
ASTM_DEPTH_LABELS = ("13–25 mm", ">25–50 mm", ">50 mm")
ASTM_DIAMETER_LABELS = ("100–200 mm", "200–450 mm", "450–750 mm")
# X1.17.1.2: wider than 750 mm, area / 0.5 m2 gives an equivalent number of potholes,
# Medium if the depth is 25 mm or less, High if more.
ASTM_LARGE_UNIT_MM2 = 0.5e6


def astm_grade(depth_mm: float, diameter_mm: float) -> Dict[str, Any]:
    """
    One ASTM D6433 grade for a known depth and average diameter.

    Returns {grade: "L"|"M"|"H"|None, cell: (row, col)|None, rule, reason}. grade is None
    where the standard gives none: shallower than 13 mm or narrower than 100 mm is not a
    pothole in Table X1.1.
    """
    if depth_mm < ASTM_DEPTH_EDGES_MM[0]:
        return {"grade": None, "cell": None, "rule": "below table",
                "reason": f"{depth_mm:.0f} mm is shallower than the 13 mm minimum of Table X1.1"}
    if diameter_mm < ASTM_DIAMETER_EDGES_MM[0]:
        return {"grade": None, "cell": None, "rule": "below table",
                "reason": f"{diameter_mm:.0f} mm across is narrower than the 100 mm minimum of Table X1.1"}
    if diameter_mm > ASTM_DIAMETER_EDGES_MM[-1]:
        g = "M" if depth_mm <= ASTM_DEPTH_EDGES_MM[1] else "H"
        n_eq = np.pi * (diameter_mm / 2.0) ** 2 / ASTM_LARGE_UNIT_MM2
        return {"grade": g, "cell": None, "rule": "X1.17.1.2",
                "reason": f"wider than 750 mm: {n_eq:.1f} equivalent potholes, "
                          f"{'Medium (depth <= 25 mm)' if g == 'M' else 'High (depth > 25 mm)'}",
                "equivalent_count": round(float(n_eq), 1)}
    row = 0 if depth_mm <= ASTM_DEPTH_EDGES_MM[1] else 1 if depth_mm <= ASTM_DEPTH_EDGES_MM[2] else 2
    col = 0 if diameter_mm <= ASTM_DIAMETER_EDGES_MM[1] else 1 if diameter_mm <= ASTM_DIAMETER_EDGES_MM[2] else 2
    g = ASTM_TABLE_X1_1[row][col]
    return {"grade": g, "cell": (row, col), "rule": "Table X1.1",
            "reason": f"{ASTM_DEPTH_LABELS[row]} deep, {ASTM_DIAMETER_LABELS[col]} across"}


# ── Grading any photo: probabilities from measured errors ─────────────────────
# Every photo is graded. How sure the grade is depends on how wrong the measurements can be,
# and that is MEASURED, not assumed (scripts/build_severity_error_model.py):
#   depth error, photos like the training set   PothRGBD held-out sessions
#   depth error, any other photo                the Fan stereo frames, a camera never seen
#   angular-size error                          the Fan point clouds
# Each measurement is resampled from its error distribution N_SAMPLES times; the share of
# samples landing in each grade is that grade's probability.
#
# WIDTH. A pothole's width needs the camera's distance, and no single-photo model estimates
# that reliably across cameras (scripts/eval_camera_distance.py: MoGe-3, Depth-Anything-V2
# metric and Depth Pro are each off by a factor that changes from 0.5x to 10x between
# cameras). So the depth x width grade of Table X1.1 is given only when the distance is
# supplied; every photo always gets the DEPTH grade: ASTM's depth bands, 13–25 / >25–50 /
# >50 mm, the basis of the simplified Low / Medium / High that many agencies use.
#
# DECISION. The reported grade minimises the expected cost of the project's safety policy:
# reporting a pothole one band milder than it is costs three times reporting it one band
# worse (the same 3:1 rule the served depth cut-offs were chosen with).
ERROR_MODEL_PATH = os.path.join(PROJECT_DIR, "ml_models", "metric", "severity_error_model.json")
N_SAMPLES = 4000
GRADING_SEED = 20261013                             # fixed, so the same photo always gets the same probabilities
COST_UNDER = 3.0
DEPTH_GRADES = ("below 13 mm", "L", "M", "H")      # band 0 = shallower than ASTM's 13 mm minimum
GRADES_2D = ("not rated", "L", "M", "H")
_ERROR_MODEL = None


def error_model() -> Dict[str, Any]:
    global _ERROR_MODEL
    if _ERROR_MODEL is None:
        with open(ERROR_MODEL_PATH, encoding="utf-8") as fh:
            _ERROR_MODEL = json.load(fh)
    return _ERROR_MODEL


def _sample(dist: Dict[str, Any], n: int, rng) -> np.ndarray:
    """Inverse-CDF sample from 101 stored quantiles."""
    return np.interp(rng.uniform(0, 100, n), np.linspace(0, 100, 101), dist["q"])


def _decide(probs: np.ndarray) -> int:
    """Grade index minimising expected safety cost (under-reporting costs COST_UNDER per band)."""
    t = np.arange(len(probs))
    cost = [float(np.sum(probs * np.where(g < t, COST_UNDER * (t - g), g - t))) for g in t]
    return int(np.argmin(cost))


def _label(grade: str) -> str:
    return ASTM_GRADE_NAMES.get(grade, "below 13 mm" if grade.startswith("below") else "not rated")


def _depth_band(true_mm: np.ndarray) -> np.ndarray:
    return ((true_mm >= ASTM_DEPTH_EDGES_MM[0]).astype(int) + (true_mm > ASTM_DEPTH_EDGES_MM[1])
            + (true_mm > ASTM_DEPTH_EDGES_MM[2]))


def depth_samples(depth_mm: float, familiar: bool, rng) -> Tuple[np.ndarray, str]:
    """Plausible true depths for a predicted one: prediction minus a sampled measured error."""
    key = "in_range" if familiar else "other_camera"
    err = _sample(error_model()["depth_error_mm"][key], N_SAMPLES, rng)
    return np.clip(depth_mm - err, 0.0, None), key


def compute_depth_grade(depth_mm: float, familiar: bool, rng) -> Dict[str, Any]:
    """ASTM depth-band grade with probabilities; given for every photo."""
    true, key = depth_samples(depth_mm, familiar, rng)
    probs = np.bincount(_depth_band(true), minlength=4) / N_SAMPLES
    g = _decide(probs)
    return {
        "basis": "depth",
        "decision": DEPTH_GRADES[g] if g > 0 else None,
        "decision_label": ASTM_GRADE_NAMES.get(DEPTH_GRADES[g], "below 13 mm (not a pothole by ASTM)"),
        "confidence": float(probs[g]),
        "most_likely_label": _label(DEPTH_GRADES[int(np.argmax(probs))]),
        "most_likely_p": float(probs.max()),
        "probabilities": {DEPTH_GRADES[i]: round(float(p), 3) for i, p in enumerate(probs)},
        "row_probabilities": [round(float(p), 3) for p in probs[1:]],
        "depth_mm": round(depth_mm, 1),
        "likely_depth_mm": [round(float(v), 1) for v in np.percentile(true, [10, 90])],
        "error_from": key,
    }


def _grade_2d(true_mm: np.ndarray, width_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised Table X1.1 + X1.17.1.2: grade index (0 not rated, 1 L, 2 M, 3 H) and cell index (-1 none)."""
    row = np.where(true_mm <= ASTM_DEPTH_EDGES_MM[1], 0, np.where(true_mm <= ASTM_DEPTH_EDGES_MM[2], 1, 2))
    col = np.where(width_mm <= ASTM_DIAMETER_EDGES_MM[1], 0, np.where(width_mm <= ASTM_DIAMETER_EDGES_MM[2], 1, 2))
    table = np.array([[{"L": 1, "M": 2, "H": 3}[g] for g in r] for r in ASTM_TABLE_X1_1])
    grade = table[row, col]
    cell = row * 3 + col
    wide = width_mm > ASTM_DIAMETER_EDGES_MM[-1]
    grade = np.where(wide, np.where(true_mm <= ASTM_DEPTH_EDGES_MM[1], 2, 3), grade)
    cell = np.where(wide, -1, cell)
    unrated = (true_mm < ASTM_DEPTH_EDGES_MM[0]) | (width_mm < ASTM_DIAMETER_EDGES_MM[0])
    return np.where(unrated, 0, grade), np.where(unrated, -1, cell)


def compute_astm_d6433_severity(depth_mm: float, angular_size: Optional[float], camera_distance_mm: Optional[float],
                                familiar: bool, rng) -> Optional[Dict[str, Any]]:
    """
    Full ASTM D6433 grade (depth x width, Table X1.1) with probabilities, or None when the
    width cannot be known (no camera distance supplied, or MoGe unavailable).
    """
    if not camera_distance_mm or not angular_size:
        return None
    true, key = depth_samples(depth_mm, familiar, rng)
    width = angular_size * camera_distance_mm / _sample(error_model()["angular_size_ratio"], N_SAMPLES, rng)
    grade, cell = _grade_2d(true, width)
    probs = np.bincount(grade, minlength=4) / N_SAMPLES
    cells = np.bincount(cell[cell >= 0], minlength=9) / N_SAMPLES
    g = _decide(probs)
    point = astm_grade(depth_mm, angular_size * camera_distance_mm)
    return {
        "basis": "depth x width",
        "decision": GRADES_2D[g] if g > 0 else None,
        "decision_label": ASTM_GRADE_NAMES.get(GRADES_2D[g], "not rated by Table X1.1"),
        "confidence": float(probs[g]),
        "most_likely_label": _label(GRADES_2D[int(np.argmax(probs))]),
        "most_likely_p": float(probs.max()),
        "probabilities": {GRADES_2D[i]: round(float(p), 3) for i, p in enumerate(probs)},
        "cell_probabilities": [round(float(p), 3) for p in cells],
        "share_wider_than_750mm": round(float((width > ASTM_DIAMETER_EDGES_MM[-1]).mean()), 3),
        "point_estimate": point,
        "width_mm": round(angular_size * camera_distance_mm, 0),
        "likely_width_mm": [round(float(v), 0) for v in np.percentile(width, [10, 90])],
        "error_from": key,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 2: Response category from a published risk matrix (2A), learned consequence (2B)
# ──────────────────────────────────────────────────────────────────────────────
# Risk = likelihood x consequence, each 1-5, with the response categories and times of the
# UK code of practice "Well-managed Highway Infrastructure" as published by Essex Highways
# (hazard.py). Likelihood depends on WHERE the pothole is (traffic, wheel path), which a
# photo cannot show, so it is an input (--likelihood, default 3 "Possible"). Consequence
# comes from the pothole: the team's learned model when one has been adopted (2B,
# scripts/fit_priority_model.py), otherwise a provisional rule anchored on the code's own
# investigatory level (hazard.consequence_from_measurements).
DEFAULT_LIKELIHOOD = 3


def compute_risk_matrix(depth_mm: float, familiar: bool, width: Optional[Dict[str, Any]], car_drop_mm: Optional[float],
                        features: Dict[str, float], likelihood: int, route: str, surface: str, rng) -> Dict[str, Any]:
    """
    Method 2. `width` is the depth x width grade (for its width) or None. The probability of
    reaching the investigatory level uses the same measured depth error as Method 1, and the
    width (when known) must also reach the level's minimum across.
    """
    level = hazard.INVESTIGATORY[surface]
    true, _ = depth_samples(depth_mm, familiar, rng)
    reach = true >= level["depth_mm"]
    if width is not None:
        w = width["width_mm"] / _sample(error_model()["angular_size_ratio"], N_SAMPLES, rng)
        reach &= w >= level["across_mm"]
    p_inv = float(reach.mean())
    learned = hazard.learned_consequence(features)
    cons = learned or hazard.consequence_from_measurements(p_inv, car_drop_mm)
    risk = hazard.risk_assessment(likelihood, cons["score"], route)
    return {**risk, "consequence_rule": cons["rule"], "consequence_source": "learned" if learned else "provisional",
            "p_investigatory": round(p_inv, 3), "investigatory_level": {"surface": surface, **level},
            "likelihood_given": likelihood != DEFAULT_LIKELIHOOD}


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 3: Wheel drop by tyre (3A, physics), learned hazard ranking (3B)
# ──────────────────────────────────────────────────────────────────────────────
# How far each kind of wheel falls into the pothole, from rolling a circle of the tyre's
# radius over the predicted cross-section (hazard.py). Validated against measured 3D on the
# PothRGBD test sessions (scripts/eval_wheel_drop.py). Needs the photo's scale, so a camera
# distance; without one only the upper bound min(deepest point, radius) is given. The
# learned ranking (3B) is the team's pairwise "which is more dangerous" judgements, fitted
# with a Bradley-Terry model (scripts/fit_hazard_pairs.py), shown once adopted.
def compute_wheel_drop(relief_mm: Optional[np.ndarray], mask: np.ndarray, depth_mm: float,
                       angular: Optional[float], camera_distance_mm: Optional[float],
                       features: Dict[str, float]) -> Dict[str, Any]:
    out = hazard.pothole_hazard(relief_mm, mask, depth_mm, angular, camera_distance_mm)
    out["learned"] = hazard.learned_hazard(features)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal Pipeline Feature Extractor
# ──────────────────────────────────────────────────────────────────────────────
def extract_pothole_features_cvpr(image_path: str, camera_distance_mm: Optional[float] = None,
                                  likelihood: int = DEFAULT_LIKELIHOOD, route: str = "local",
                                  surface: str = "carriageway") -> Optional[Dict[str, Any]]:
    """
    Runs the full multi-modal pipeline on an image and extracts all signals.

    Every measured pothole is graded. A photo like the measured training photos is graded
    with the error measured on held-out training-camera photos; any other photo with the
    larger error measured on a camera the network never saw, and says so. Nothing is
    invented: a pothole with no depth estimate is reported as not measured.
    `camera_distance_mm` is the distance from the camera to the road at the pothole (e.g. a
    fixed mounting height); with it, the width, the full depth x width grade and the wheel
    drops are added. `likelihood`, `route` and `surface` describe where the pothole is, for
    the risk matrix (Method 2).
    """
    rng = np.random.default_rng(GRADING_SEED)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_bgr.shape[:2]

    # 1. Pothole Segmentation
    masks = segmentation.get_all_masks(image_path)
    if not masks:
        return {
            "image_bgr": image_bgr,
            "image_rgb": image_rgb,
            "masks": [],
            "potholes": [],
        }

    # 2. Metric Depth Geometry
    metric_geo = metric_severity.geometry_for_image(image_bgr, masks)

    # 3. Water Segmentation
    water_full_map = water_segmenter.water_map(image_bgr)

    pothole_records = []
    for idx, mask in enumerate(masks):
        # Physical metric prediction
        metric = metric_severity.predict(metric_geo, idx) if metric_geo else None

        if metric is None or metric.get("depthMm") is None:
            pothole_records.append({"id": idx + 1, "mask": mask, "measured": False,
                                    "status": "no depth estimate (depth model unavailable or the pothole is too small)"})
            continue
        depth_mm = float(metric["depthMm"])
        interval_mm = float(metric.get("intervalMm") or 0.0)
        in_dist = metric.get("inDistribution")          # True / False, or None when MoGe is absent
        source = metric.get("source")
        familiar = in_dist is True
        if familiar:
            status = "like the training photos: training-camera error"
        elif in_dist is False:
            status = "unlike the training photos: graded with second-camera error"
        else:
            status = "not checked (MoGe unavailable): graded with second-camera error"
        # Angular size = MoGe diameter / MoGe camera distance. MoGe's scale cancels in the
        # ratio (accurate within 25% on every Fan pothole); its millimetres are never used.
        angular = (float(metric["diameterMm"]) / float(metric["cameraDistanceMm"])
                   if metric.get("diameterMm") and metric.get("cameraDistanceMm") else None)

        # Water detection
        water_res = water_detection.detect_water(
            image_rgb, mask, water_segmentation_map=water_full_map
        )

        # DINOv2 foundation features: the same ratio, threshold and patch floor as the app's
        # semantic override (semantic_config; still uncalibrated hand-set defaults).
        dinov2_res = {"variance_ratio": None, "patches": 0, "illusion_warning": False}
        if foundation_features.HAS_DINOV2:
            try:
                feats = foundation_features.extract_foundation_features(image_rgb, mask)
                if feats:
                    ratio = feats.get(semantic_config.RATIO_KEY)
                    patches = int(feats.get("dinov2_patch_count", 0) or 0)
                    dinov2_res.update(variance_ratio=ratio, patches=patches, illusion_warning=bool(
                        ratio is not None and patches >= semantic_config.MIN_PATCHES
                        and ratio < semantic_config.DOWNGRADE_RATIO))
            except Exception as e:
                print(f"  DINOv2 features failed: {e}")

        # Temporal deterioration: None when unavailable, never a made-up default
        temporal_res = {"edge_sharpness": None, "crack_texture": None}
        try:
            curv = features.extract_curvature_features(mask)
            age = temporal_analysis.estimate_pothole_age(mask, image_rgb, curv)
            if age:
                temporal_res["edge_sharpness"] = age.get("edge_sharpness")
                temporal_res["crack_texture"] = age.get("crack_texture_score")
        except Exception as e:
            print(f"  age estimation failed: {e}")

        # ── Compute the 3 CVPR Formulations ──
        relief = metric_geo.get("relief") if metric_geo else None
        geo_mask = metric_geo["masks"][idx] if metric_geo else mask        # same scale as the relief map
        deepest = (float(np.max(relief[geo_mask > 0])) if relief is not None and (geo_mask > 0).any() else None)
        feats = hazard.feature_row(depth_mm, deepest, water_res.get("segmenter_coverage"),
                                   temporal_res["edge_sharpness"], temporal_res["crack_texture"],
                                   dinov2_res["variance_ratio"], float((mask > 0).mean()))
        depth_grade = compute_depth_grade(depth_mm, familiar, rng)                     # Method 1, always
        astm_2d = compute_astm_d6433_severity(depth_mm, angular, camera_distance_mm, familiar, rng)  # with width
        used = astm_2d or depth_grade
        wheel = compute_wheel_drop(relief, geo_mask, depth_mm, angular, camera_distance_mm, feats)  # Method 3
        car = next(t for t in wheel["tyres"] if t["id"] == "car")
        risk = compute_risk_matrix(depth_mm, familiar, astm_2d, car["drop_mm"], feats,               # Method 2
                                   likelihood, route, surface, rng)

        pothole_records.append({
            "id": idx + 1,
            "mask": mask,
            "measured": True,
            "status": status,
            "familiar": familiar,
            "depth_mm": depth_mm,
            "interval_mm": interval_mm,
            "angular_size": angular,
            "camera_distance_given_mm": camera_distance_mm,
            "in_distribution": in_dist,
            "out_of_range_features": metric.get("outOfRange", []),
            "depth_source": source,
            "read_from": metric.get("readFrom"),
            "water": water_res,
            "dinov2": dinov2_res,
            "temporal": temporal_res,
            "method1_depth_grade": depth_grade,
            "method1_astm_2d": astm_2d,
            "grade_used": used,
            "features": feats,
            "method2_risk_matrix": risk,
            "method3_wheel_drop": wheel,
        })

    return {
        "image_bgr": image_bgr,
        "image_rgb": image_rgb,
        "masks": masks,
        "water_full_map": water_full_map,
        "metric_geo": metric_geo,
        "potholes": pothole_records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Comprehensive Visual Diagnostic Panel Generator
# ──────────────────────────────────────────────────────────────────────────────
def render_cvpr_diagnostic_panel(
    data: Dict[str, Any],
    image_name: str,
    output_path: str,
):
    """Renders the comprehensive CVPR multi-metric visual comparison panel."""
    image_rgb = data["image_rgb"]
    potholes = data["potholes"]
    h, w = image_rgb.shape[:2]

    # Representative pothole: the first one with a depth estimate (masks come largest first)
    measured = [p for p in potholes if p.get("measured")]
    if not measured:
        print(f"No measured potholes to visualize in {image_name}")
        return
    rep = measured[0]

    fig = plt.figure(figsize=(26, 13), dpi=150)
    fig.patch.set_facecolor("#0b0f19")  # Dark scientific slate theme

    # Grid Spec: 2 Rows x 4 Columns
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0], hspace=0.28, wspace=0.22)

    # ──────────────────────────────────────────────────────────────────────────
    # Column 1, Row 1: Original Image + YOLO Segmentation
    # ──────────────────────────────────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    orig_vis = image_rgb.copy()
    for p in potholes:
        contours, _ = cv2.findContours(p["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_vis, contours, -1, (16, 185, 129), 2)
        ys, xs = np.nonzero(p["mask"])
        if len(xs) > 0:
            cv2.rectangle(orig_vis, (xs.min(), ys.min()), (xs.max(), ys.max()), (234, 179, 8), 2)
            cv2.putText(
                orig_vis,
                f"Pothole #{p['id']}",
                (xs.min(), max(ys.min() - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (234, 179, 8),
                2,
            )

    ax_orig.imshow(orig_vis)
    ax_orig.set_title("1. Pothole Detection (YOLOv8-Seg)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax_orig.axis("off")

    # ──────────────────────────────────────────────────────────────────────────
    # Column 2, Row 1: Multimodal Defect Overlay (Depth + Water)
    # ──────────────────────────────────────────────────────────────────────────
    ax_overlay = fig.add_subplot(gs[0, 1])
    overlay = image_rgb.copy().astype(np.float32)

    mask = rep["mask"]
    # Translucent orange pothole base
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([249, 115, 22]) * 0.5

    # Water inside pothole highlighted in cyan
    water_map = data.get("water_full_map")
    if water_map is not None:
        puddle_in_pothole = water_map & (mask > 0)
        if puddle_in_pothole.any():
            overlay[puddle_in_pothole] = overlay[puddle_in_pothole] * 0.35 + np.array([6, 182, 212]) * 0.65

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Contours
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (249, 115, 22), 2)

    ax_overlay.imshow(overlay)
    ax_overlay.set_title("2. Multimodal Surface Overlay\n(Cyan: Submerged Water | Orange: Rim)",
                         color="#f8fafc", fontsize=12, fontweight="bold")
    ax_overlay.axis("off")

    # ──────────────────────────────────────────────────────────────────────────
    # Column 3, Row 1: Method 1 - ASTM D6433 2D Bivariate Matrix
    # ──────────────────────────────────────────────────────────────────────────
    ax_astm = fig.add_subplot(gs[0, 2])
    ax_astm.set_facecolor("#111827")
    ax_astm.axis("off")

    used = rep["grade_used"]
    depth_grade = rep["method1_depth_grade"]
    two_d = rep["method1_astm_2d"]
    sev_colors = {"L": "#10b981", "M": "#f59e0b", "H": "#ef4444"}

    # Header Card: the grade in use and how probable it is
    basis = "DEPTH x WIDTH (TABLE X1.1)" if two_d else "DEPTH BANDS (WIDTH UNKNOWN)"
    ax_astm.text(0.5, 0.95, f"METHOD 1: ASTM D6433, {basis}", color="#38bdf8", fontsize=10.5, fontweight="bold", ha="center")
    label = used["decision_label"].upper()
    if used["decision_label"] == used["most_likely_label"]:
        badge = f"GRADE: {label}  ({used['confidence']:.0%})"
    else:
        badge = f"GRADE: {label} (safety choice) · most likely {used['most_likely_label']} {used['most_likely_p']:.0%}"
    ax_astm.text(0.5, 0.86, badge, color="#ffffff", fontsize=11.5 if len(badge) > 30 else 12.5, fontweight="bold",
                 ha="center", bbox=dict(boxstyle="round,pad=0.45", facecolor=sev_colors.get(used["decision"], "#64748b"),
                                        edgecolor="none"))

    # Probability bar: how the plausible true values split across grades
    names = list(used["probabilities"].keys())
    x0 = 0.05
    for name in names:
        p = used["probabilities"][name]
        if p <= 0:
            continue
        ax_astm.add_patch(Rectangle((x0, 0.72), 0.90 * p, 0.05, facecolor=sev_colors.get(name, "#475569"),
                                    edgecolor="#0b0f19", linewidth=1.0))
        if p >= 0.07:
            short = ASTM_GRADE_NAMES.get(name, "below 13" if name.startswith("below") else "not rated")
            ax_astm.text(x0 + 0.45 * p, 0.745, f"{short} {p:.0%}", color="#ffffff", fontsize=7.5,
                         ha="center", va="center", fontweight="bold")
        x0 += 0.90 * p

    # Table X1.1 as a probability heat map. With the width known, each cell is shaded by its
    # probability; without it, a whole depth row is shaded by that row's probability.
    grid_y = np.linspace(0.62, 0.38, 4)
    grid_x = np.linspace(0.26, 0.94, 4)
    cell_p = (np.array(two_d["cell_probabilities"]).reshape(3, 3) if two_d
              else np.repeat(np.array(depth_grade["row_probabilities"])[:, None], 3, axis=1))
    for r in range(3):
        ax_astm.text(grid_x[0] - 0.02, (grid_y[r] + grid_y[r + 1]) / 2, ASTM_DEPTH_LABELS[r],
                     color="#94a3b8", fontsize=7.5, ha="right", va="center")
        for c in range(3):
            cell_val = ASTM_TABLE_X1_1[r][c]
            p = float(cell_p[r, c])
            ax_astm.add_patch(Rectangle(
                (grid_x[c], grid_y[r + 1]), grid_x[c + 1] - grid_x[c], grid_y[r] - grid_y[r + 1],
                facecolor=sev_colors[cell_val], alpha=0.12 + 0.83 * min(1.0, p / max(cell_p.max(), 1e-9)),
                edgecolor="#1e293b", linewidth=1.0))
            ax_astm.text((grid_x[c] + grid_x[c + 1]) / 2, (grid_y[r] + grid_y[r + 1]) / 2,
                         f"{cell_val}\n{p:.0%}" if two_d else cell_val,
                         color="#ffffff", fontsize=8, ha="center", va="center")
        if not two_d:
            ax_astm.text(grid_x[-1] + 0.01, (grid_y[r] + grid_y[r + 1]) / 2, f"{depth_grade['row_probabilities'][r]:.0%}",
                         color="#e2e8f0", fontsize=8, ha="left", va="center", fontweight="bold")
    for c in range(3):
        ax_astm.text((grid_x[c] + grid_x[c + 1]) / 2, grid_y[0] + 0.025, ASTM_DIAMETER_LABELS[c],
                     color="#94a3b8", fontsize=7.5, ha="center")

    lo, hi = depth_grade["likely_depth_mm"]
    width_line = (f"Width         : {two_d['width_mm']:.0f} mm (likely {two_d['likely_width_mm'][0]:.0f}–"
                  f"{two_d['likely_width_mm'][1]:.0f})" if two_d else
                  "Width         : unknown without the camera distance (--camera-distance-mm)")
    ax_astm.text(
        0.03, 0.31,
        f"Est. depth    : {depth_grade['depth_mm']} mm (likely {lo:.0f}–{hi:.0f})\n"
        f"{width_line}\n"
        f"Error model   : {'training-camera photos' if used['error_from'] == 'in_range' else 'a camera never seen (Fan)'}\n"
        f"Decision      : 3:1 safety cost (under-reporting penalised)",
        color="#cbd5e1", fontsize=7.8, va="top", linespacing=1.4, fontfamily="monospace"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Column 4, Row 1: Method 2 - UK risk matrix
    # ──────────────────────────────────────────────────────────────────────────
    _draw_risk_matrix(fig.add_subplot(gs[0, 3]), rep["method2_risk_matrix"])

    # ──────────────────────────────────────────────────────────────────────────
    # Columns 1-2, Row 2: Method 3 - wheel drop by tyre
    # ──────────────────────────────────────────────────────────────────────────
    _draw_wheel_drop(fig, gs[1, 0:2], rep)

    # ──────────────────────────────────────────────────────────────────────────
    # Columns 3-4, Row 2: Comprehensive Comparative Table
    # ──────────────────────────────────────────────────────────────────────────
    _draw_comparison(fig.add_subplot(gs[1, 2:4]), rep)

    two_d = rep["method1_astm_2d"]
    lo, hi = rep["method1_depth_grade"]["likely_depth_mm"]
    width_txt = f"width {two_d['width_mm']:.0f} mm" if two_d else "width unknown (no camera distance)"
    plt.suptitle(
        f"RoadLens severity evaluation: {image_name[:70]}\n"
        f"Photo {rep['status']}  |  depth {rep['depth_mm']:.1f} mm (likely {lo:.0f}–{hi:.0f})  |  {width_txt}",
        color="#f8fafc", fontsize=13, fontweight="bold", y=0.98
    )

    fig.subplots_adjust(top=0.89, bottom=0.06, left=0.03, right=0.97, hspace=0.30, wspace=0.20)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[OK] Visual Panel Saved: {output_path}")


CATEGORY_COLORS = {"Observation": "#10b981", "D3": "#f59e0b", "D3A": "#f97316", "D2": "#ef4444", "D1": "#b91c1c"}


def _draw_risk_matrix(ax, risk):
    """Method 2: the 5 x 5 matrix coloured by response category, the pothole's cell outlined."""
    ax.set_facecolor("#111827")
    ax.axis("off")
    ax.text(0.5, 0.95, "METHOD 2: UK RISK MATRIX", color="#38bdf8", fontsize=11, fontweight="bold", ha="center")
    ax.text(0.5, 0.905, "Well-managed Highway Infrastructure (Essex Highways 2026)", color="#64748b",
            fontsize=7.5, ha="center")
    cat = risk["category"]
    ax.text(0.5, 0.83, f"{cat}: respond within {risk['response']}", color="#ffffff", fontsize=11.5,
            fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=CATEGORY_COLORS.get(cat, "#64748b"), edgecolor="none"))

    x0, y0, cw, ch = 0.30, 0.30, 0.12, 0.084             # consequence across, likelihood up
    for li in range(1, 6):
        for ci in range(1, 6):
            c = hazard.response_for(li * ci, risk["route"])["category"]
            x, y = x0 + (ci - 1) * cw, y0 + (li - 1) * ch
            here = li == risk["likelihood"] and ci == risk["consequence"]
            ax.add_patch(Rectangle((x, y), cw, ch, facecolor=CATEGORY_COLORS.get(c, "#64748b"),
                                   alpha=0.95 if here else 0.30, edgecolor="#ffffff" if here else "#0b0f19",
                                   linewidth=2.5 if here else 0.8))
            ax.text(x + cw / 2, y + ch / 2, str(li * ci), color="#ffffff", fontsize=8 if not here else 10,
                    fontweight="bold" if here else "normal", ha="center", va="center")
    for i in range(5):
        ax.text(x0 + (i + 0.5) * cw, y0 - 0.03, str(i + 1), color="#94a3b8", fontsize=8, ha="center", va="top")
        ax.text(x0 - 0.015, y0 + (i + 0.5) * ch, str(i + 1), color="#94a3b8", fontsize=8, ha="right", va="center")
    ax.text(x0 + 2.5 * cw, y0 - 0.075, "consequence", color="#94a3b8", fontsize=8, ha="center", va="top")
    ax.text(x0 - 0.10, y0 + 2.5 * ch, "likelihood", color="#94a3b8", fontsize=8, ha="center", va="center", rotation=90)

    lvl = risk["investigatory_level"]
    lik = (f"{risk['likelihood']} {risk['likelihood_label']}" + ("" if risk["likelihood_given"]
                                                                   else " (default; set --likelihood)"))
    cons_src = ("learned from team ratings (2B)" if risk["consequence_source"] == "learned"
                else "provisional rule (2A) until 2B is adopted")
    ax.text(0.03, 0.15,
            f"Likelihood  : {lik}\n"
            f"Consequence : {risk['consequence']} {risk['consequence_label']}, {cons_src}\n"
            f"P(reaches {lvl['depth_mm']} mm investigatory level) = {risk['p_investigatory']:.0%}",
            color="#cbd5e1", fontsize=7.6, va="top", linespacing=1.45, fontfamily="monospace")


TYRE_COLORS = {"bicycle": "#a3e635", "scooter": "#22d3ee", "auto_rickshaw": "#facc15", "motorcycle": "#fb923c",
               "car": "#f472b6", "suv": "#c084fc", "truck": "#94a3b8"}


def _draw_wheel_drop(fig, cell, rep):
    """Method 3: the cross-section with three wheels at their lowest point, and the drop of every tyre."""
    wd = rep["method3_wheel_drop"]
    outer = fig.add_subplot(cell)
    outer.set_facecolor("#111827")
    outer.axis("off")
    title = "METHOD 3: WHEEL DROP BY TYRE (PHYSICS, VALIDATED ON MEASURED 3D)"
    outer.text(0.5, 0.95, title, color="#38bdf8", fontsize=12, fontweight="bold", ha="center")
    sub = (f"scale from width {wd['width_mm']:.0f} mm (angular size x camera distance)" if wd["scaled"] else
           "no camera distance: the photo has no horizontal scale, so only the upper bound is known")
    outer.text(0.5, 0.885, sub, color="#64748b", fontsize=8.5, ha="center")

    # a spacer row on top keeps the plots clear of the titles above
    sub_gs = cell.subgridspec(2, 2, height_ratios=[0.22, 1.0], width_ratios=[1.15, 1.0], wspace=0.25, hspace=0.0)
    ax_x = fig.add_subplot(sub_gs[1, 0])
    ax_b = fig.add_subplot(sub_gs[1, 1])
    for a in (ax_x, ax_b):
        a.set_facecolor("#0f172a")
        a.tick_params(colors="#94a3b8", labelsize=8)
        for s in a.spines.values():
            s.set_color("#334155")

    tyres = wd["tyres"]
    patch = wd.get("patch")
    if patch is not None:
        r = np.asarray(patch["relief_mm"], np.float64)
        m = (r > 0).astype(np.uint8)
        car = next(t for t in tyres if t["id"] == "car")
        axis = car["profile_axis"] if car["profile_axis"] is not None else 0
        prof = hazard.band_profile(r, m, axis, 1)
        step = (wd["mm_per_px"] * patch["px_per_cell"]) if wd["scaled"] else None
        x = np.arange(prof.size) * (step or 1.0)
        ax_x.fill_between(x, prof, -max(5.0, -prof.min()) * 1.35, color="#475569")
        ax_x.plot(x, prof, color="#e2e8f0", lw=1.5)
        if step:
            for t in tyres:
                if t["id"] not in ("bicycle", "car", "truck"):
                    continue
                band = max(1, int(round(t["width_mm"] / step)))
                h = hazard.band_profile(r, m, axis, band)
                c = hazard.wheel_centre(h, step, t["radius_mm"])
                i = int(np.argmin(c))
                th = np.linspace(0, 2 * np.pi, 400)
                ax_x.plot(x[i] + t["radius_mm"] * np.sin(th), c[i] - t["radius_mm"] * np.cos(th),
                          color=TYRE_COLORS[t["id"]], lw=1.6, label=f"{t['label']} (R {t['radius_mm']:.0f} mm)")
            ax_x.legend(loc="lower right", fontsize=7, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
            ax_x.set_xlabel("along the travel line (mm)", color="#94a3b8", fontsize=8)
        else:
            ax_x.set_xlabel("along the travel line (pixels: no scale)", color="#94a3b8", fontsize=8)
        ax_x.set_ylim(-max(5.0, -prof.min()) * 1.35, max(5.0, -prof.min()) * 0.6)
        ax_x.set_xlim(x[0], x[-1])
        ax_x.set_ylabel("height (mm, exaggerated)", color="#94a3b8", fontsize=8)
        ax_x.set_title("cross-section through the deepest point", color="#e2e8f0", fontsize=9)
    else:
        ax_x.text(0.5, 0.5, "no relief map", color="#94a3b8", ha="center", va="center", transform=ax_x.transAxes)

    names = [t["label"] for t in tyres]
    y = np.arange(len(tyres))[::-1]
    bounds = [t["drop_bound_mm"] for t in tyres]
    ax_b.barh(y, bounds, color="none", edgecolor="#64748b", hatch="///", lw=0.8, label="upper bound (no scale)")
    if wd["scaled"]:
        drops = [t["drop_mm"] for t in tyres]
        ax_b.barh(y, drops, color=[TYRE_COLORS[t["id"]] for t in tyres], label="wheel drop")
        for yy, d in zip(y, drops):
            ax_b.text(d + 0.5, yy, f"{d:.0f} mm", color="#e2e8f0", fontsize=8, va="center")
    else:
        for yy, d in zip(y, bounds):
            ax_b.text(d + 0.5, yy, f"≤ {d:.0f} mm", color="#e2e8f0", fontsize=8, va="center")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(names, fontsize=8, color="#e2e8f0")
    ax_b.set_xlim(0, max(bounds) * 1.25 + 1)
    ax_b.set_xlabel("drop of the wheel centre (mm)", color="#94a3b8", fontsize=8)
    ax_b.set_title("how far each wheel falls in", color="#e2e8f0", fontsize=9)
    ax_b.legend(loc="lower right", fontsize=7, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
    if wd.get("learned"):
        ln = wd["learned"]
        outer.text(0.5, 0.02, f"3B learned hazard: more dangerous than {ln['percentile']:.0f}% of labelled potholes "
                   f"({ln['rule']})", color="#fbbf24", fontsize=8.5, ha="center")


def _draw_comparison(ax_comp, rep):
    """The readings of one pothole side by side."""
    ax_comp.set_facecolor("#111827")
    ax_comp.axis("off")
    ax_comp.set_title("Cross-Framework Method Comparison", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)

    dg, two_d = rep["method1_depth_grade"], rep["method1_astm_2d"]
    risk, wd = rep["method2_risk_matrix"], rep["method3_wheel_drop"]
    car = next(t for t in wd["tyres"] if t["id"] == "car")
    bike = next(t for t in wd["tyres"] if t["id"] == "bicycle")
    served = metric_severity.decide(rep["depth_mm"], relief_depth.meta()) if rep["depth_source"] == "relief" else "—"
    drop_txt = (f"car {car['drop_mm']:.0f} mm, bicycle {bike['drop_mm']:.0f} mm" if wd["scaled"]
                else f"up to {car['drop_bound_mm']:.0f} mm (no scale)")
    learned = wd.get("learned")
    table_data = [
        ["Served depth cut-offs (app)", f"{rep['depth_mm']:.1f} mm", "Depth only, safety-weighted cut-offs", served],
        ["Method 1: ASTM depth bands", f"{rep['depth_mm']:.1f} mm", "Probabilistic, measured error",
         f"{dg['decision_label']} (most likely {dg['most_likely_label']} {dg['most_likely_p']:.0%})"],
        ["Method 1: ASTM Table X1.1", f"{rep['depth_mm']:.1f} x {two_d['width_mm']:.0f} mm" if two_d else "needs camera distance",
         "Depth x width, probabilistic", f"{two_d['decision_label']} ({two_d['confidence']:.0%})" if two_d else "—"],
        ["Method 2: UK risk matrix", f"L{risk['likelihood']} x C{risk['consequence']} = {risk['score']}",
         "Published code; C " + ("learned (2B)" if risk["consequence_source"] == "learned" else "provisional (2A)"),
         f"{risk['category']} ({risk['response']})"],
        ["Method 3: wheel drop (3A)", drop_txt, "Tyre rolled over cross-section",
         "scale from camera distance" if wd["scaled"] else "bound only"],
        ["Method 3: learned rank (3B)", f"{learned['percentile']:.0f}th percentile" if learned else "awaiting team labels",
         "Bradley-Terry on pairwise judgements", "adopted" if learned else "—"],
    ]

    table = ax_comp.table(
        cellText=table_data,
        colLabels=["Framework", "Physical Measurement", "Decision Basis", "Output Label"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1.0, 1.7)

    for (r_i, c_i), cell in table.get_celld().items():
        cell.set_edgecolor("#1e293b")
        if r_i == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#38bdf8", fontweight="bold")
        else:
            cell.set_facecolor("#0f172a" if r_i % 2 == 0 else "#1e293b")
            cell.set_text_props(color="#e2e8f0")


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI & Batch Runner
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="CVPR Pavement Defect Evaluation Harness")
    parser.add_argument("--image", "-i", type=str, default=DEFAULT_IMAGE_PATH,
                        help="Path to a single image to analyze.")
    parser.add_argument("--folder", "-f", type=str, default=None,
                        help="Folder of images for batch evaluation.")
    parser.add_argument("--output-dir", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save output panels and json manifests.")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Maximum images to process in folder mode.")
    parser.add_argument("--camera-distance-mm", type=float, default=None,
                        help="Known camera-to-road distance at the pothole (e.g. a fixed mount). Gives a "
                             "diameter accurate to about 9%% (Fan validation) instead of a range.")
    parser.add_argument("--likelihood", type=int, choices=range(1, 6), default=DEFAULT_LIKELIHOOD,
                        help="Risk-matrix likelihood 1-5 for where the pothole is (traffic, wheel path); "
                             "default 3 'Possible'.")
    parser.add_argument("--route", choices=sorted(hazard.ROUTES), default="local",
                        help="county routes get the 2-working-day D2 response, local routes 5.")
    parser.add_argument("--surface", choices=sorted(hazard.INVESTIGATORY), default="carriageway",
                        help="investigatory level: carriageway 100 mm across / 50 mm deep, cycleway 75 / 20.")
    return parser.parse_args()


def process_single_image(image_path: str, output_dir: str, camera_distance_mm: Optional[float] = None,
                         likelihood: int = DEFAULT_LIKELIHOOD, route: str = "local",
                         surface: str = "carriageway") -> Optional[Dict[str, Any]]:
    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None

    filename = os.path.basename(image_path)
    # Roboflow names run past 100 characters, which can push the output path over Windows'
    # 260-character limit; the first 40 keep them unique enough to find.
    stem = os.path.splitext(filename)[0][:40]
    panel_out = os.path.join(output_dir, f"{stem}_cvpr_panel.png")
    json_out = os.path.join(output_dir, f"{stem}_cvpr_metrics.json")

    print(f"\nAnalyzing image: {filename}")
    data = extract_pothole_features_cvpr(image_path, camera_distance_mm, likelihood, route, surface)
    if not data or not data["potholes"]:
        print(f"No potholes detected in {filename}")
        return None

    # Render Visual Panel
    render_cvpr_diagnostic_panel(data, filename, panel_out)

    # Dump JSON Manifest
    serializable = []
    for p in data["potholes"]:
        if not p.get("measured"):
            serializable.append({"pothole_id": p["id"], "status": p["status"]})
            continue
        serializable.append({
            "pothole_id": p["id"],
            "status": p["status"],
            "in_distribution": p["in_distribution"],
            "out_of_range_features": p["out_of_range_features"],
            "depth_mm": round(p["depth_mm"], 1),
            "interval_mm": round(p["interval_mm"], 1),
            "depth_source": p["depth_source"],
            "read_from": p["read_from"],
            "angular_size": None if p["angular_size"] is None else round(p["angular_size"], 4),
            "camera_distance_given_mm": p["camera_distance_given_mm"],
            "water_detected": p["water"].get("is_water", False),
            "water_coverage": p["water"].get("segmenter_coverage"),
            "dinov2": p["dinov2"],
            "method1_depth_grade": p["method1_depth_grade"],
            "method1_astm_table_x1_1": p["method1_astm_2d"],
            "features": {k: (None if v != v else round(v, 4)) for k, v in p["features"].items()},
            "method2_risk_matrix": p["method2_risk_matrix"],
            "method3_wheel_drop": {k: v for k, v in p["method3_wheel_drop"].items() if k != "patch"},
        })

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[OK] JSON Manifest Saved: {json_out}")
    return serializable[0]


def main():
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ROADLENS CVPR SEVERITY FRAMEWORK TEST HARNESS")
    print("  Integrating:")
    print("    1. ASTM D6433 severity (depth bands; Table X1.1 with a camera distance)")
    print("    2. UK risk matrix: likelihood x consequence -> response category and time")
    print("    3. Wheel drop by tyre (physics), and the team's learned hazard ranking when adopted")
    print(f"  Target Output Directory: {output_dir}")
    print("=" * 80)

    if args.folder:
        folder = os.path.abspath(args.folder)
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(folder, ext)))
        images = sorted(list(set(images)))
        if args.limit:
            images = images[: args.limit]
        print(f"Processing batch of {len(images)} images from {folder}...")
        for img in images:
            process_single_image(img, output_dir, args.camera_distance_mm, args.likelihood, args.route, args.surface)
    else:
        image_path = os.path.abspath(args.image)
        process_single_image(image_path, output_dir, args.camera_distance_mm, args.likelihood, args.route, args.surface)


if __name__ == "__main__":
    main()

