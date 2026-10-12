"""
CVPR Research Extension: Multi-Modal Metric Defect Characterization Framework.
==============================================================================

This script implements the three proposed CVPR formulations in a standalone,
parallel test harness without modifying the production RoadLens codebase:

1. Method 1 (ASTM D6433 2D Bivariate Matrix):
   Physical defect tensor D = (depth_mm, diameter_mm, area_cm2) mapped to
   the international ASTM D6433 Pavement Condition Index 2D standard.

2. Method 2 (Two-Tier Municipal Operational Framework):
   - Tier 1: Physical Defect Class (ASTM D6433 bivariate geometry).
   - Tier 2: Municipal Maintenance Priority (Priority 1 Emergency [24-48h],
             Priority 2 Scheduled [7-14d], Priority 3 Monitoring).

3. Method 3 (Road Hazard Score - RHS: 0 to 100):
   Continuous risk-sensitive Bayesian score incorporating geometric deformation,
   water epistemic uncertainty inflation (broken Lambertian assumption),
   progression acceleration, and DINOv2 texture illusion penalties.

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
import metric_features
import metric_severity
import relief_depth
import segmentation
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
# METHOD 1: ASTM D6433 2D Bivariate Severity Matrix
# ──────────────────────────────────────────────────────────────────────────────
def compute_astm_d6433_severity(depth_mm: float, diameter_mm: float) -> Dict[str, Any]:
    """
    Evaluates severity using the international ASTM D6433 2D Pavement Condition Index standard.

    ASTM D6433 Bivariate Rule:
    --------------------------
    Depth Bands:
      - Shallow  : < 25.0 mm (< 1 inch)
      - Moderate : 25.0 mm to 50.0 mm (1 to 2 inches)
      - Deep     : >= 50.0 mm (> 2 inches)

    Diameter Bands:
      - Small    : < 200.0 mm (< 8 inches)
      - Medium   : 200.0 mm to 450.0 mm (8 to 18 inches)
      - Large    : >= 450.0 mm (> 18 inches)

    Matrix Transitions:
      - Shallow Depth (<25mm) with Large Diameter (>=450mm) -> Promoted to Moderate.
      - Moderate Depth (25-50mm) with Small Diameter (<200mm) -> Demoted to Shallow.
      - Moderate Depth (25-50mm) with Large Diameter (>=450mm) -> Promoted to Deep.
      - Deep Depth (>=50mm) with Small Diameter (<200mm) -> Demoted to Moderate.
    """
    # 1. Determine Depth Band
    if depth_mm < 25.0:
        d_band = "Shallow"
        d_idx = 0
    elif depth_mm < 50.0:
        d_band = "Moderate"
        d_idx = 1
    else:
        d_band = "Deep"
        d_idx = 2

    # 2. Determine Diameter Band
    if diameter_mm < 200.0:
        diam_band = "Small"
        diam_idx = 0
    elif diameter_mm < 450.0:
        diam_band = "Medium"
        diam_idx = 1
    else:
        diam_band = "Large"
        diam_idx = 2

    # 3. Lookup ASTM Matrix
    # Matrix rows: [Shallow, Moderate, Deep], cols: [Small, Medium, Large]
    matrix = [
        ["Shallow", "Shallow", "Moderate"],   # Shallow depth
        ["Shallow", "Moderate", "Deep"],      # Moderate depth
        ["Moderate", "Deep", "Deep"],         # Deep depth
    ]
    astm_severity = matrix[d_idx][diam_idx]

    # Explain the geometric rationale
    if d_band == astm_severity:
        rationale = f"1D depth ({d_band}) confirmed by {diam_band} diameter ({diameter_mm:.0f} mm)."
    elif d_band == "Shallow" and astm_severity == "Moderate":
        rationale = f"Promoted to Moderate: Shallow depth ({depth_mm:.1f} mm) spans a wide {diameter_mm:.0f} mm impact zone."
    elif d_band == "Moderate" and astm_severity == "Deep":
        rationale = f"Promoted to Deep: Moderate depth ({depth_mm:.1f} mm) combined with large diameter ({diameter_mm:.0f} mm) poses high suspension risk."
    elif d_band == "Moderate" and astm_severity == "Shallow":
        rationale = f"Demoted to Shallow: Moderate depth ({depth_mm:.1f} mm) is a narrow pit (<200 mm) that tires easily bridge."
    elif d_band == "Deep" and astm_severity == "Moderate":
        rationale = f"Demoted to Moderate: Deep reading ({depth_mm:.1f} mm) is an isolated narrow pocket (<200 mm)."
    else:
        rationale = f"ASTM 2D Matrix verdict: {astm_severity}."

    return {
        "severity": astm_severity,
        "depth_band": d_band,
        "diameter_band": diam_band,
        "depth_mm": round(depth_mm, 1),
        "diameter_mm": round(diameter_mm, 1),
        "cell_coords": (d_idx, diam_idx),
        "rationale": rationale,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 2: Two-Tier Municipal Operational Framework
# ──────────────────────────────────────────────────────────────────────────────
def compute_two_tier_severity(
    astm_res: Dict[str, Any],
    water_res: Dict[str, Any],
    temporal_res: Dict[str, Any],
    dinov2_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Tier 1: Physical Defect Class (ASTM D6433 bivariate geometry).
    Tier 2: Municipal Maintenance Priority based on environmental & optical risk.
    """
    tier1_class = astm_res["severity"]

    is_water = water_res.get("is_water", False)
    water_cov = water_res.get("segmenter_coverage", 0.0) or 0.0
    is_illusion = dinov2_res.get("illusion_warning", False)
    edge_sharpness = temporal_res.get("edge_sharpness", 0.5)

    reasons = []

    # Priority 1: Emergency Intervention (24-48 Hours)
    # - Any physically Deep pothole
    # - Moderate pothole filled with water (concealed depth + sub-base blowout risk)
    # - Moderate pothole with acute edge sharpness (>0.75) under high traffic/weather
    if tier1_class == "Deep":
        priority = "Priority 1 (Emergency)"
        window = "Within 24 - 48 Hours"
        color = "#ef4444"  # Red
        reasons.append("Structural defect categorized as Deep under ASTM D6433")
        if is_water:
            reasons.append(f"Submerged water trap ({water_cov:.1%} coverage) accelerating sub-base degradation")

    elif tier1_class == "Moderate" and is_water:
        priority = "Priority 1 (Emergency)"
        window = "Within 24 - 48 Hours"
        color = "#ef4444"
        reasons.append("Moderate defect submerged in water: concealed depth hazard & hydraulic pumping risk")

    elif tier1_class == "Moderate" and edge_sharpness >= 0.75:
        priority = "Priority 1 (Emergency)"
        window = "Within 48 - 72 Hours"
        color = "#f97316"  # Orange
        reasons.append(f"Acute edge sharpness ({edge_sharpness:.2f}) indicates rapid structural fracture")

    # Priority 2: Scheduled Maintenance (7-14 Days)
    # - Standard Moderate dry potholes
    # - Wide Shallow potholes
    elif tier1_class == "Moderate":
        priority = "Priority 2 (Scheduled)"
        window = "Within 7 - 14 Days"
        color = "#f59e0b"  # Amber
        reasons.append("Moderate structural defect on dry pavement; standard maintenance queue")

    elif tier1_class == "Shallow" and astm_res["diameter_band"] == "Large":
        priority = "Priority 2 (Scheduled)"
        window = "Within 14 Days"
        color = "#f59e0b"
        reasons.append("Shallow depression spanning extensive surface area (>450 mm)")

    # Priority 3: Routine Monitoring
    # - Minor shallow defects or texture artifacts
    else:
        priority = "Priority 3 (Routine Monitoring)"
        window = "Next Inspection Cycle"
        color = "#10b981"  # Emerald
        if is_illusion:
            reasons.append("DINOv2 texture check suggests smooth surface discoloration / low depth confidence")
        else:
            reasons.append("Minor shallow deformation (<25 mm); does not disrupt vehicular travel")

    return {
        "tier1_physical_class": tier1_class,
        "tier2_priority": priority,
        "dispatch_window": window,
        "badge_color": color,
        "priority_reasons": reasons,
    }


# ──────────────────────────────────────────────────────────────────────────────
# METHOD 3: Road Hazard Score (RHS: 0–100) via Risk-Sensitive Bayesian Model
# ──────────────────────────────────────────────────────────────────────────────
def compute_road_hazard_score(
    depth_mm: float,
    diameter_mm: float,
    water_res: Dict[str, Any],
    temporal_res: Dict[str, Any],
    dinov2_res: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes a continuous Road Hazard Score (0-100) grounded in physical risk:
      RHS = clip(R_geometry + U_water + S_progression - P_illusion, 0, 100)

    Theoretical Basis:
      - Water violates the Lambertian reflectance assumption. Since d_true >= d_observed,
        water coverage inflates epistemic risk and effective depth.
      - DINOv2 self-supervised patch variance penalizes 2D surface stains.
    """
    # 1. Geometric Risk (0 to 60 points)
    # Depth component: 0 to 40 pts (saturates at 60 mm)
    depth_pts = min(40.0, 40.0 * (depth_mm / 60.0))
    # Diameter component: 0 to 20 pts (saturates at 600 mm)
    diam_pts = min(20.0, 20.0 * (diameter_mm / 600.0))
    r_geometry = depth_pts + diam_pts

    # 2. Water Epistemic Uncertainty & Lower-Bound Inflation (0 to 25 points)
    # Water coverage c_w in [0, 1], probability p_w in [0, 1]
    is_water = water_res.get("is_water", False)
    c_water = water_res.get("segmenter_coverage", 0.0) or 0.0
    p_water = water_res.get("water_probability", 0.0) or 0.0

    if is_water:
        u_water = 25.0 * (c_water * 0.7 + p_water * 0.3)
        # Effective lower-bound depth: true bottom is deeper than the water surface
        depth_effective_mm = depth_mm + (18.0 * c_water)
    else:
        u_water = 0.0
        depth_effective_mm = depth_mm

    # 3. Progression Acceleration (0 to 15 points)
    edge_sharpness = temporal_res.get("edge_sharpness", 0.5)
    crack_score = temporal_res.get("crack_texture", 0.0)
    s_progression = (edge_sharpness * 10.0) + (crack_score * 5.0)

    # 4. DINOv2 Texture Illusion Penalty (0 to 30 points)
    ratio = dinov2_res.get("variance_ratio")
    p_illusion = 0.0
    if ratio is not None and ratio < 0.70:
        # Ratio < 0.70 indicates smooth 2D texture (e.g. oil or fresh patch)
        p_illusion = 30.0 * (1.0 - (ratio / 0.70))

    # Composite Score
    rhs_raw = r_geometry + u_water + s_progression - p_illusion
    rhs_final = float(np.clip(rhs_raw, 0.0, 100.0))

    # Risk Band Categorization
    if rhs_final >= 65.0:
        risk_level = "Critical Risk"
        risk_color = "#ef4444"  # Red
    elif rhs_final >= 35.0:
        risk_level = "Moderate Risk"
        risk_color = "#f59e0b"  # Amber
    else:
        risk_level = "Low Risk"
        risk_color = "#10b981"  # Emerald

    return {
        "rhs_score": round(rhs_final, 1),
        "risk_level": risk_level,
        "risk_color": risk_color,
        "effective_depth_mm": round(depth_effective_mm, 1),
        "breakdown": {
            "geometry_pts": round(r_geometry, 1),
            "water_uncertainty_pts": round(u_water, 1),
            "progression_pts": round(s_progression, 1),
            "illusion_penalty_pts": round(p_illusion, 1),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal Pipeline Feature Extractor
# ──────────────────────────────────────────────────────────────────────────────
def extract_pothole_features_cvpr(image_path: str) -> Optional[Dict[str, Any]]:
    """Runs the full multi-modal pipeline on an image and extracts all signals."""
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

        # Default fallback values if physical models are absent
        depth_mm = metric.get("depthMm", 22.0) if metric else 22.0
        diameter_mm = metric.get("diameterMm", 320.0) if metric else 320.0
        interval_mm = metric.get("intervalMm", 14.0) if metric else 14.0
        cam_dist_mm = metric.get("cameraDistanceMm", 950.0) if metric else 950.0
        in_dist = metric.get("inDistribution", True) if metric else True
        source = metric.get("source", "relief") if metric else "estimated"

        # Water detection
        water_res = water_detection.detect_water(
            image_rgb, mask, water_segmentation_map=water_full_map
        )

        # DINOv2 foundation features
        dinov2_res = {"variance_ratio": None, "illusion_warning": False}
        if foundation_features.HAS_DINOV2:
            try:
                feats = foundation_features.extract_foundation_features(image_rgb, mask)
                if feats:
                    dinov2_res["variance_ratio"] = feats.get("dinov2_variance_ratio")
                    dinov2_res["illusion_warning"] = (
                        dinov2_res["variance_ratio"] is not None
                        and dinov2_res["variance_ratio"] < 0.70
                    )
            except Exception:
                pass

        # Temporal deterioration
        temporal_res = {"edge_sharpness": 0.5, "crack_texture": 0.2}
        try:
            curv = features.extract_curvature_features(mask)
            age = temporal_analysis.estimate_pothole_age(mask, image_rgb, curv)
            if age:
                temporal_res["edge_sharpness"] = age.get("edge_sharpness", 0.5)
                temporal_res["crack_texture"] = age.get("crack_texture_score", 0.2)
        except Exception:
            pass

        # ── Compute the 3 CVPR Formulations ──
        # Method 1
        astm = compute_astm_d6433_severity(depth_mm, diameter_mm)
        # Method 2
        two_tier = compute_two_tier_severity(astm, water_res, temporal_res, dinov2_res)
        # Method 3
        rhs = compute_road_hazard_score(depth_mm, diameter_mm, water_res, temporal_res, dinov2_res)

        pothole_records.append({
            "id": idx + 1,
            "mask": mask,
            "depth_mm": depth_mm,
            "interval_mm": interval_mm,
            "diameter_mm": diameter_mm,
            "camera_dist_mm": cam_dist_mm,
            "in_distribution": in_dist,
            "depth_source": source,
            "water": water_res,
            "dinov2": dinov2_res,
            "temporal": temporal_res,
            "method1_astm": astm,
            "method2_two_tier": two_tier,
            "method3_rhs": rhs,
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

    # Select representative pothole (most severe or largest)
    if not potholes:
        print(f"No potholes to visualize in {image_name}")
        return
    rep = potholes[0]  # Primary detected defect

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

    astm = rep["method1_astm"]
    sev_colors = {"Shallow": "#10b981", "Moderate": "#f59e0b", "Deep": "#ef4444"}

    # Header Card
    ax_astm.text(0.5, 0.94, "METHOD 1: ASTM D6433 2D MATRIX", color="#38bdf8", fontsize=11, fontweight="bold", ha="center")
    badge_color = sev_colors.get(astm["severity"], "#64748b")
    ax_astm.text(0.5, 0.83, f"VERDICT: {astm['severity'].upper()}", color="#ffffff", fontsize=13, fontweight="bold",
                 ha="center", bbox=dict(boxstyle="round,pad=0.5", facecolor=badge_color, edgecolor="none"))

    # Matrix Visual Grid (3x3)
    grid_y = np.linspace(0.66, 0.36, 4)
    grid_x = np.linspace(0.12, 0.92, 4)
    labels_d = ["Shallow (<25)", "Mod (25-50)", "Deep (>=50)"]
    labels_w = ["Small (<200)", "Med (200-450)", "Large (>=450)"]
    mat_cells = [
        ["Shallow", "Shallow", "Moderate"],
        ["Shallow", "Moderate", "Deep"],
        ["Moderate", "Deep", "Deep"],
    ]

    active_r, active_c = astm["cell_coords"]

    for r in range(3):
        for c in range(3):
            cell_val = mat_cells[r][c]
            is_active = (r == active_r and c == active_c)
            c_box = sev_colors[cell_val]
            alpha = 0.95 if is_active else 0.20
            rect = Rectangle(
                (grid_x[c], grid_y[r + 1]),
                grid_x[c + 1] - grid_x[c],
                grid_y[r] - grid_y[r + 1],
                facecolor=c_box,
                alpha=alpha,
                edgecolor="#38bdf8" if is_active else "#1e293b",
                linewidth=2.5 if is_active else 1.0,
            )
            ax_astm.add_patch(rect)
            ax_astm.text(
                (grid_x[c] + grid_x[c + 1]) / 2,
                (grid_y[r] + grid_y[r + 1]) / 2,
                cell_val[:3],
                color="#ffffff" if is_active else "#94a3b8",
                fontsize=8.5,
                fontweight="bold" if is_active else "normal",
                ha="center",
                va="center",
            )

    ax_astm.text(
        0.05, 0.22,
        f"Measured Depth   : {astm['depth_mm']} mm ({astm['depth_band']})\n"
        f"Est. Diameter    : {astm['diameter_mm']} mm ({astm['diameter_band']})\n"
        f"Matrix Rationale : {astm['rationale']}",
        color="#cbd5e1", fontsize=8.5, va="top", linespacing=1.4, fontfamily="monospace"
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Column 4, Row 1: Method 2 - Two-Tier Municipal Operational System
    # ──────────────────────────────────────────────────────────────────────────
    ax_tier = fig.add_subplot(gs[0, 3])
    ax_tier.set_facecolor("#111827")
    ax_tier.axis("off")

    two_tier = rep["method2_two_tier"]
    ax_tier.text(0.5, 0.94, "METHOD 2: TWO-TIER MUNICIPAL SYSTEM", color="#38bdf8", fontsize=11, fontweight="bold", ha="center")

    # Tier 1 Banner
    t1_box = dict(boxstyle="square,pad=0.3", facecolor="#1e293b", edgecolor="#334155")
    ax_tier.text(0.5, 0.81, f"TIER 1 (Physical Defect): {two_tier['tier1_physical_class']}", color="#f1f5f9",
                 fontsize=10, fontweight="bold", ha="center", bbox=t1_box)

    # Tier 2 Banner
    t2_color = two_tier["badge_color"]
    t2_box = dict(boxstyle="round,pad=0.5", facecolor=t2_color, edgecolor="none")
    ax_tier.text(0.5, 0.65, f"TIER 2: {two_tier['tier2_priority'].upper()}", color="#ffffff",
                 fontsize=11.5, fontweight="bold", ha="center", bbox=t2_box)

    ax_tier.text(0.5, 0.52, f"Action Window: {two_tier['dispatch_window']}", color="#fbbf24",
                 fontsize=10, fontweight="bold", ha="center")

    # Triggering Factors
    import textwrap
    wrapped_reasons = []
    for r in two_tier["priority_reasons"]:
        wrapped_reasons.extend(textwrap.wrap(f"• {r}", width=48))
    reasons_fmt = "\n".join(wrapped_reasons)
    ax_tier.text(0.05, 0.40, f"Triggering Policy Drivers:\n{reasons_fmt}",
                 color="#cbd5e1", fontsize=8.5, va="top", linespacing=1.4, fontfamily="monospace")

    # ──────────────────────────────────────────────────────────────────────────
    # Columns 1-2, Row 2: Method 3 - Continuous Road Hazard Score (RHS)
    # ──────────────────────────────────────────────────────────────────────────
    ax_rhs = fig.add_subplot(gs[1, 0:2])
    ax_rhs.set_facecolor("#111827")
    ax_rhs.axis("off")

    rhs = rep["method3_rhs"]
    ax_rhs.text(0.5, 0.94, "METHOD 3: CONTINUOUS ROAD HAZARD SCORE (RHS 0-100)", color="#38bdf8",
                fontsize=12, fontweight="bold", ha="center")

    # Gauge Progress Bar
    bar_y = 0.76
    ax_rhs.barh(bar_y, 0.90, left=0.05, height=0.08, color="#1e293b")
    rhs_ratio = rhs["rhs_score"] / 100.0
    ax_rhs.barh(bar_y, 0.90 * rhs_ratio, left=0.05, height=0.08, color=rhs["risk_color"])

    ax_rhs.text(0.5, bar_y, f"RHS: {rhs['rhs_score']} / 100  ({rhs['risk_level'].upper()})",
                color="#ffffff", fontsize=11, fontweight="bold", ha="center", va="center")

    # Breakdown bars
    b = rhs["breakdown"]
    sub_cues = [
        ("Geometric Deformation", b["geometry_pts"], 60.0, "#3b82f6"),
        ("Water Epistemic Uncertainty", b["water_uncertainty_pts"], 25.0, "#06b6d4"),
        ("Progression Acceleration", b["progression_pts"], 15.0, "#f97316"),
        ("DINOv2 Texture Penalty", -b["illusion_penalty_pts"], 30.0, "#a855f7"),
    ]

    y_subs = [0.55, 0.42, 0.29, 0.16]
    for (label, val, max_v, col), y_s in zip(sub_cues, y_subs):
        val_str = f"+{val:.1f}" if val >= 0 else f"{val:.1f}"
        ax_rhs.text(0.05, y_s, f"{label:28s} [{val_str:>5s} pts]", color="#94a3b8", fontsize=9.5, fontfamily="monospace")
        ax_rhs.barh(y_s, 0.40, left=0.55, height=0.06, color="#1e293b")
        fill_w = 0.40 * (abs(val) / max_v)
        ax_rhs.barh(y_s, max(fill_w, 0.005), left=0.55, height=0.06, color=col)

    ax_rhs.text(0.05, 0.04, f"Effective Submerged Depth (Lower Bound): {rhs['effective_depth_mm']} mm",
                color="#38bdf8", fontsize=10, fontweight="bold")

    # ──────────────────────────────────────────────────────────────────────────
    # Columns 3-4, Row 2: Comprehensive Comparative Table
    # ──────────────────────────────────────────────────────────────────────────
    ax_comp = fig.add_subplot(gs[1, 2:4])
    ax_comp.set_facecolor("#111827")
    ax_comp.axis("off")
    ax_comp.set_title("Cross-Framework Method Comparison", color="#f8fafc", fontsize=12, fontweight="bold", pad=8)

    table_data = [
        ["Baseline Relief (1D Cutoff)", f"{rep['depth_mm']} mm", "Single 1D Depth Cutoff", rep["method1_astm"]["depth_band"]],
        ["Method 1: ASTM D6433 (2D Matrix)", f"{rep['depth_mm']} mm x {rep['diameter_mm']} mm", "Bivariate Depth x Diameter", rep["method1_astm"]["severity"]],
        ["Method 2: Two-Tier Municipal", f"Tier 1: {rep['method2_two_tier']['tier1_physical_class']}", "Physical Class + Repair Priority", rep["method2_two_tier"]["tier2_priority"]],
        ["Method 3: Road Hazard Score", f"RHS: {rep['method3_rhs']['rhs_score']} / 100", "Bayesian Optical Uncertainty", rep["method3_rhs"]["risk_level"]],
    ]

    table = ax_comp.table(
        cellText=table_data,
        colLabels=["Framework", "Physical Measurement", "Decision Basis", "Output Label"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.8)

    for (r_i, c_i), cell in table.get_celld().items():
        cell.set_edgecolor("#1e293b")
        if r_i == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#38bdf8", fontweight="bold")
        else:
            cell.set_facecolor("#0f172a" if r_i % 2 == 0 else "#1e293b")
            cell.set_text_props(color="#e2e8f0")

    plt.suptitle(
        f"RoadLens CVPR Multimodal Evaluation: {image_name}\n"
        f"Input Geometry: Depth {rep['depth_mm']} mm (±{rep['interval_mm']} mm) | Diameter {rep['diameter_mm']} mm | Camera Distance {rep['camera_dist_mm']} mm",
        color="#f8fafc", fontsize=14, fontweight="bold", y=0.98
    )

    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.03, right=0.97, hspace=0.30, wspace=0.20)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[OK] Visual Panel Saved: {output_path}")


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
    return parser.parse_args()


def process_single_image(image_path: str, output_dir: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return None

    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]
    panel_out = os.path.join(output_dir, f"{stem}_cvpr_panel.png")
    json_out = os.path.join(output_dir, f"{stem}_cvpr_metrics.json")

    print(f"\nAnalyzing image: {filename}")
    data = extract_pothole_features_cvpr(image_path)
    if not data or not data["potholes"]:
        print(f"No potholes detected in {filename}")
        return None

    # Render Visual Panel
    render_cvpr_diagnostic_panel(data, filename, panel_out)

    # Dump JSON Manifest
    serializable = []
    for p in data["potholes"]:
        serializable.append({
            "pothole_id": p["id"],
            "depth_mm": p["depth_mm"],
            "interval_mm": p["interval_mm"],
            "diameter_mm": p["diameter_mm"],
            "camera_dist_mm": p["camera_dist_mm"],
            "water_detected": p["water"].get("is_water", False),
            "water_coverage": p["water"].get("segmenter_coverage"),
            "method1_astm_d6433": p["method1_astm"],
            "method2_two_tier": p["method2_two_tier"],
            "method3_road_hazard_score": p["method3_rhs"],
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
    print("    1. ASTM D6433 2D Bivariate Matrix (Depth x Diameter)")
    print("    2. Two-Tier Municipal Operational Framework (Defect x Dispatch Priority)")
    print("    3. Continuous Road Hazard Score (RHS 0-100 with Water Uncertainty)")
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
            process_single_image(img, output_dir)
    else:
        image_path = os.path.abspath(args.image)
        process_single_image(image_path, output_dir)


if __name__ == "__main__":
    main()

