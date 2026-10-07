import base64
import csv
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import existing pipeline methods
from segmentation import get_all_masks
from inference import (
    get_depth_map,
    extract_ml_features,
    load_ml_artifacts,
    SEVERITY_MAP,
)
from features import extract_depth_features
from classifier import classify_severity
import semantic_config

# Optional imports for new modules (graceful degradation)
try:
    from features import extract_all_geometry_features, extract_curvature_features
    _HAS_GEOMETRY = True
except ImportError:
    _HAS_GEOMETRY = False

try:
    from water_detection import detect_water as _detect_water
    import water_config
    _HAS_WATER_DETECTION = True
except ImportError:
    _HAS_WATER_DETECTION = False

# Cue 7 — CLIPSeg open-vocabulary water prior. Optional, gated off by default.
try:
    import semantic_water
    _HAS_SEMANTIC_WATER = semantic_water.HAS_CLIPSEG
except Exception:
    _HAS_SEMANTIC_WATER = False

# Cue 8 — intrinsic-decomposition specular residual. Optional, gated off.
try:
    import intrinsic_cues
    _HAS_INTRINSIC = intrinsic_cues.HAS_INTRINSIC
except Exception:
    _HAS_INTRINSIC = False

# Learned water segmenter — the primary water decision when its weights exist.
try:
    import water_segmenter
    _HAS_WATER_SEGMENTER = water_segmenter.HAS_WATER_SEGMENTER
except Exception:
    _HAS_WATER_SEGMENTER = False

try:
    from temporal_analysis import estimate_pothole_age, predict_severity_progression
    _HAS_TEMPORAL = True
except ImportError:
    _HAS_TEMPORAL = False

# Measured-label depth model: MoGe geometry + a regressor fitted on RealSense
# labels. When present it supplies the headline severity (metric_severity.SEVERITY_MODE);
# the rule + pseudo-label vote is kept in the response for comparison.
try:
    import metric_severity
    _HAS_METRIC = metric_severity.HAS_METRIC
except Exception:
    _HAS_METRIC = False

app = FastAPI(title="Pothole Detection API")
PROJECT_ROOT = Path(__file__).resolve().parent
ML_RESULTS_DIR = PROJECT_ROOT / "ml_results"

# /analyze runs in FastAPI's threadpool (it is a plain `def`), so the event loop
# stays free for /healthz and /insights while a scan computes. The shared YOLO,
# Depth-Anything, DINOv2 and CLIPSeg singletons are not safe to drive from two
# threads at once, so scans are serialised: concurrent requests queue here
# instead of interleaving forward passes on the same model objects.
_PIPELINE_LOCK = threading.Lock()


def _get_allowed_origins() -> List[str]:
    """Read allowed frontend origins from env; keep local defaults for development."""
    configured = os.getenv("FRONTEND_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if origins:
        return origins

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


ALLOWED_ORIGINS = _get_allowed_origins()

SEVERITY_WEIGHT = {
    "No Pothole": 0,
    "No pothole": 0,
    "Shallow": 1,
    "Moderate": 2,
    "Deep": 3,
}

SEVERITY_COLORS_BGR = {
    "No Pothole": (0, 255, 0),
    "Shallow": (0, 255, 255),
    "Moderate": (0, 165, 255),
    "Deep": (0, 0, 255),
}

GRAPH_CATEGORY_MAP = [
    ("accuracy_", "Accuracy"),
    ("ablation_", "Ablation"),
    ("calibration_", "Calibration"),
    ("bootstrap_", "Bootstrap"),
    ("confusion_matrix_", "Confusion Matrix"),
    ("feature_correlation", "Feature Relationships"),
    ("kmeans_", "Clustering"),
    ("learning_curve_", "Learning Curves"),
    ("shap_", "SHAP"),
    ("tsne_", "Embeddings"),
    ("umap_", "Embeddings"),
]

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image(image: np.ndarray) -> str:
    """Encode an OpenCV BGR image as a base64 JPEG string."""
    ret, buffer = cv2.imencode(".jpg", image)
    if not ret:
        return ""
    base64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def compute_depth_slice(
    mask: np.ndarray, depth_map: np.ndarray, water_detected: bool = False, severity: str = "Moderate", image_gray: Optional[np.ndarray] = None
) -> Optional[Dict[str, object]]:
    """Compute a horizontal depth cross-section through a pothole centroid.

    Returns the raw sample arrays for charting: actual depth values along the
    slice plus a quadratic road-surface extrapolation from outside the mask.
    """
    if mask is None or depth_map is None or np.sum(mask) == 0:
        return None

    h, w = mask.shape[:2]

    # Normalize depth to [0, 1]
    d = depth_map.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        d = (d - d_min) / (d_max - d_min)
    else:
        return None

    # Find centroid
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())

    # Bounding box for the mask region
    x_min, x_max = int(xs.min()), int(xs.max())
    p_width = max(1, x_max - x_min)
    margin = int(p_width * 0.8)  # extend beyond the pothole

    # Sample range along the horizontal slice (y = cy)
    sample_start = max(0, x_min - margin)
    sample_end = min(w - 1, x_max + margin)

    slice_points = []
    outside_xs = []
    outside_depths = []

    for px in range(sample_start, sample_end + 1):
        depth_val = float(d[cy, px])
        in_mask = bool(mask[cy, px] > 0)
        slice_points.append({
            "x": px - sample_start,  # relative position
            "actualDepth": round(depth_val, 4),
            "inMask": in_mask,
        })
        if not in_mask:
            outside_xs.append(float(px - sample_start))
            outside_depths.append(depth_val)

    if len(outside_xs) < 4 or len(slice_points) < 5:
        return None

    # Fit quadratic to road surface outside the mask
    try:
        road_coeffs = np.polyfit(outside_xs, outside_depths, 2)
        for pt in slice_points:
            x_val = float(pt["x"])
            pt["roadSurface"] = round(
                float(road_coeffs[0] * x_val**2 + road_coeffs[1] * x_val + road_coeffs[2]), 4
            )
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use mean of outside depths as flat road surface
        road_mean = float(np.mean(outside_depths))
        for pt in slice_points:
            pt["roadSurface"] = round(road_mean, 4)

    # If water is detected, extrapolate the hidden depth using boundary curvature (the visible walls)
    predicted_bowl_depth = None
    if water_detected:
        # Find the mask boundaries in our slice
        mask_indices = [i for i, pt in enumerate(slice_points) if pt["inMask"]]
        if len(mask_indices) > 10:
            m_start = mask_indices[0]
            m_end = mask_indices[-1]
            m_width = m_end - m_start
            
            # Use the outer 20% of the pothole as the "visible walls" before it hits the flat water
            wall_margin = max(3, int(m_width * 0.20))
            
            wall_xs = []
            wall_depths = []
            
            # Include points just outside the mask (road edge) and just inside (pothole lip)
            for i in range(max(0, m_start - 5), min(len(slice_points), m_start + wall_margin)):
                wall_xs.append(slice_points[i]["x"])
                wall_depths.append(slice_points[i]["actualDepth"])
                
            for i in range(max(0, m_end - wall_margin), min(len(slice_points), m_end + 5)):
                wall_xs.append(slice_points[i]["x"])
                wall_depths.append(slice_points[i]["actualDepth"])
                
            try:
                # Fit a parabola specifically to the walls to predict the bottom
                wall_coeffs = np.polyfit(wall_xs, wall_depths, 2)
                
                a = wall_coeffs[0]
                b = wall_coeffs[1]
                c = wall_coeffs[2]
                
                # Check if parabola is concave-up (bowl shape). In MiDaS, lower depth = further away = deeper.
                # So a bowl should curve downwards (a > 0).
                # If 'a' is near 0 or negative, the depth model completely failed to see the walls due to water.
                if a < 1e-5:
                    # GEOMETRIC FALLBACK: Infer 3D depth from 2D width + ML Consensus
                    w_px = slice_points[m_end]["x"] - slice_points[m_start]["x"]
                    if w_px > 0:
                        mid_x = (slice_points[m_start]["x"] + slice_points[m_end]["x"]) / 2.0
                        
                        # Scale the extrapolated depth based on the non-depth ML classifiers
                        if severity == "Shallow":
                            multiplier = 0.0002
                        elif severity == "Deep":
                            multiplier = 0.0012
                        else:
                            multiplier = 0.0006
                            
                        d_max = w_px * multiplier
                        a_fallback = (4 * d_max) / (w_px ** 2)
                        
                        for pt in slice_points:
                            x_val = float(pt["x"])
                            # Parabola centered at mid_x, dropping by d_max
                            predicted = a_fallback * (x_val - mid_x)**2 + (pt["roadSurface"] - d_max)
                            
                            predicted = min(predicted, pt["roadSurface"])
                            if pt["inMask"]:
                                predicted = min(predicted, pt["actualDepth"])
                            pt["predictedDepth"] = round(predicted, 4)
                else:
                    # Use the 3D derived wall curvature
                    for pt in slice_points:
                        x_val = float(pt["x"])
                        predicted = float(a * x_val**2 + b * x_val + c)
                        
                        predicted = min(predicted, pt["roadSurface"])
                        if pt["inMask"]:
                            predicted = min(predicted, pt["actualDepth"])
                        pt["predictedDepth"] = round(predicted, 4)
                        
                # Calculate predicted bowl depth
                pred_gaps = [abs(pt["roadSurface"] - pt.get("predictedDepth", pt["actualDepth"])) for pt in slice_points if pt["inMask"]]
                if pred_gaps:
                    predicted_bowl_depth = round(max(pred_gaps), 4)
            except (np.linalg.LinAlgError, ValueError):
                pass
                
    # If prediction failed or no water, predicted depth is just actual depth
    for pt in slice_points:
        if "predictedDepth" not in pt:
            pt["predictedDepth"] = pt["actualDepth"]

    # Compute actual bowl depth (gap between road surface and actual/water surface)
    bowl_depth = 0.0
    for pt in slice_points:
        if pt["inMask"]:
            gap = abs(pt["roadSurface"] - pt["actualDepth"])
            bowl_depth = max(bowl_depth, gap)

    # SfS FALLBACK (For Dry, Flat MiDaS Failures)
    if not water_detected and bowl_depth < 0.02 and image_gray is not None:
        try:
            from shape_from_shading import reconstruct_depth_sfs
            sfs_height = reconstruct_depth_sfs(image_gray, mask)
            if sfs_height is not None:
                # sfs_height is [0,1] normalized height. h=0 is bottom, h=1 is road.
                scale = 0.05 if severity == "Moderate" else (0.1 if severity == "Deep" else 0.02)
                for pt in slice_points:
                    if pt["inMask"]:
                        x_val = float(pt["x"])
                        px = int(x_val + sample_start)
                        # The drop from road surface
                        sfs_depth_drop = (1.0 - float(sfs_height[cy, px])) * scale
                        predicted = pt["roadSurface"] - sfs_depth_drop
                        
                        # SfS provides high-frequency micro-geometry. 
                        # We merge it with actualDepth to preserve whatever macro shape exists.
                        predicted = min(predicted, pt["actualDepth"]) 
                        pt["predictedDepth"] = round(predicted, 4)
                        
                # Recalculate predicted bowl depth
                pred_gaps = [abs(pt["roadSurface"] - pt.get("predictedDepth", pt["actualDepth"])) for pt in slice_points if pt["inMask"]]
                if pred_gaps:
                    predicted_bowl_depth = round(max(pred_gaps), 4)
        except Exception:
            pass

    return {
        "slicePoints": slice_points,
        "sliceY": cy,
        "sliceStartX": sample_start,
        "sliceEndX": sample_end,
        "bowlDepth": round(bowl_depth, 4),
        "predictedBowlDepth": predicted_bowl_depth,
    }


def build_depth_annotated_image(
    original_image: np.ndarray,
    depth_map: np.ndarray,
    mask_records: List[Tuple[np.ndarray, Dict[str, object]]],
) -> np.ndarray:
    """Build an annotated depth overlay with contours, markers, and slice lines."""
    h, w = original_image.shape[:2]
    canvas = original_image.copy()

    # Normalize depth to 0-255 for contouring
    d = depth_map.astype(np.float32)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max > d_min:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros((h, w), dtype=np.uint8)

    # Depth colormap for the overlay inside pothole masks
    depth_colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)

    for mask, pothole in mask_records:
        severity = str(pothole.get("consensusSeverity", "No Pothole"))
        sev_color = SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))

        # Blend depth colormap inside the mask region
        mask_bool = mask > 0
        blended = cv2.addWeighted(original_image, 0.35, depth_colored, 0.65, 0)
        canvas[mask_bool] = blended[mask_bool]

        # Draw iso-depth contour lines inside each pothole
        masked_depth = d_norm.copy()
        masked_depth[~mask_bool] = 0
        n_levels = 6
        depth_vals = d_norm[mask_bool]
        if len(depth_vals) > 0:
            lo, hi = int(depth_vals.min()), int(depth_vals.max())
            if hi > lo:
                step = max(1, (hi - lo) // n_levels)
                for level in range(lo + step, hi, step):
                    binary = (masked_depth >= level).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(
                        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(
                        canvas, contours, -1, (255, 255, 255), 1, cv2.LINE_AA
                    )

        # Min/Max depth markers
        ys, xs = np.where(mask_bool)
        if len(ys) > 0:
            depth_vals_f = d[mask_bool]
            min_idx = int(np.argmin(depth_vals_f))
            max_idx = int(np.argmax(depth_vals_f))

            # Min depth marker (cyan diamond)
            mn_x, mn_y = int(xs[min_idx]), int(ys[min_idx])
            cv2.drawMarker(
                canvas, (mn_x, mn_y), (255, 255, 0),
                cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "MIN", (mn_x + 8, mn_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA,
            )

            # Max depth marker (red diamond)
            mx_x, mx_y = int(xs[max_idx]), int(ys[max_idx])
            cv2.drawMarker(
                canvas, (mx_x, mx_y), (0, 0, 255),
                cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "MAX", (mx_x + 8, mx_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
            )

        # Draw the horizontal slice line through centroid
        depth_profile = pothole.get("depthProfile")
        if depth_profile:
            sy = depth_profile["sliceY"]
            sx1 = depth_profile["sliceStartX"]
            sx2 = depth_profile["sliceEndX"]
            cv2.line(canvas, (sx1, sy), (sx2, sy), (0, 255, 255), 1, cv2.LINE_AA)
            # Small endpoint markers
            cv2.drawMarker(
                canvas, (sx1, sy), (0, 255, 255),
                cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA,
            )
            cv2.drawMarker(
                canvas, (sx2, sy), (0, 255, 255),
                cv2.MARKER_TILTED_CROSS, 6, 1, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, "SLICE", (sx1 + 4, sy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA,
            )

        # Pothole ID label
        bbox_obj = pothole.get("bbox")
        if bbox_obj:
            cv2.putText(
                canvas,
                f"P{pothole['id']} — {severity}",
                (int(bbox_obj["x1"]), max(15, int(bbox_obj["y1"]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, sev_color, 2, cv2.LINE_AA,
            )

    return canvas


def draw_labeled_bbox(
    image: np.ndarray,
    bbox: tuple,
    lines: list,
    box_color=(0, 0, 255),
) -> None:
    """Draw a bounding box and a readable multi-line label block."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    if not lines:
        return

    line_h = 20
    text_w = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0] for text in lines)
    pad = 8
    box_w = text_w + 2 * pad
    box_h = line_h * len(lines) + 2 * pad

    x_text = max(5, min(x1, image.shape[1] - box_w - 5))
    y_text = y1 - box_h - 8
    if y_text < 5:
        y_text = min(image.shape[0] - box_h - 5, y2 + 8)

    overlay = image.copy()
    cv2.rectangle(overlay, (x_text, y_text), (x_text + box_w, y_text + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for idx, text in enumerate(lines):
        y = y_text + pad + 14 + idx * line_h
        cv2.putText(
            image,
            text,
            (x_text + pad, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def normalize_severity(label: str) -> str:
    if label == "No pothole":
        return "No Pothole"
    return label


def severity_rank(label: str) -> int:
    return SEVERITY_WEIGHT.get(label, 0)


def majority_vote(verdicts: List[str]) -> Tuple[str, int, int]:
    if not verdicts:
        return "No Pothole", 0, 0

    counts = {v: verdicts.count(v) for v in set(verdicts)}
    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1], severity_rank(item[0])),
        reverse=True,
    )
    winner = ranked[0][0]
    return winner, counts[winner], len(verdicts)


def semantic_override(consensus: str, rule_severity: str, ratio, n_patches: int
                      ) -> Tuple[str, str, bool]:
    """
    Apply the DINOv2 material check to a voted verdict.

    Returns (final_severity, semantic_verdict, illusion_warning). Two-way: an
    interior much smoother than the road around it downgrades an over-reported
    Deep; a much more heterogeneous one raises an under-reported Shallow. Abstains
    when the interior covers too few patches for a variance to mean anything.
    """
    if ratio is None or ratio != ratio:  # missing or NaN
        return consensus, "unavailable", False
    if n_patches < semantic_config.MIN_PATCHES:
        return consensus, "abstained_sparse_patches", False
    if ratio < semantic_config.DOWNGRADE_RATIO:
        if consensus == "Deep" or rule_severity == "Deep":
            return "Shallow", "downgraded_texture_illusion", True
        return consensus, "smooth_interior_no_action", False
    if ratio > semantic_config.UPGRADE_RATIO:
        if normalize_severity(consensus) == "Shallow":
            return "Moderate", "raised_heterogeneous_interior", False
        return consensus, "heterogeneous_interior_confirms", False
    return consensus, "inconclusive", False


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def feature_bundle(depth_features: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not depth_features:
        return {
            "pothole_area": 0,
            "max_depth": 0.0,
            "mean_depth": 0.0,
            "depth_std": 0.0,
            "depth_range": 0.0,
            "p90_depth": 0.0,
            "local_depth_contrast": 0.0,
        }

    return {
        "pothole_area": int(depth_features.get("area", 0)),
        "max_depth": float(depth_features.get("max_depth", 0.0)),
        "mean_depth": float(depth_features.get("mean_depth", 0.0)),
        "depth_std": float(depth_features.get("depth_std", 0.0)),
        "depth_range": float(depth_features.get("depth_range", 0.0)),
        "p90_depth": float(depth_features.get("p90_depth", 0.0)),
        "local_depth_contrast": float(depth_features.get("local_depth_contrast", 0.0)),
    }


def build_schematic_image(image_shape: Tuple[int, int, int], potholes: List[Dict[str, object]]) -> np.ndarray:
    h, w = image_shape[:2]
    schematic = np.zeros((h, w, 3), dtype=np.uint8)

    for pothole in potholes:
        bbox = pothole.get("bbox")
        if not bbox:
            continue

        severity = str(pothole.get("consensusSeverity", "No Pothole"))
        color = SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])

        cv2.rectangle(schematic, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            schematic,
            f"P{pothole.get('id', 0)} {severity}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        schematic,
        "Schematic: Per-pothole bounding boxes and severity",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return schematic


def _safe_results_file(file_name: str) -> Path:
    """Resolve a file under ml_results while preventing path traversal."""
    if not file_name or "/" in file_name or "\\" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    target = (ML_RESULTS_DIR / file_name).resolve()
    results_root = ML_RESULTS_DIR.resolve()

    if not str(target).startswith(str(results_root)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _water_cue_status(first: Dict) -> Dict[str, str]:
    """
    Per-cue state for the water ensemble.

    Six cues always run; two are optional and default off. Reporting which of
    the eight actually voted keeps an absent witness distinguishable from one
    that voted against — the same discipline the module status strip applies at
    phase level.
    """
    wa = first.get("waterAnalysis") or {}

    def optional(enabled: bool, importable: bool, score_present: bool) -> str:
        if not importable:
            return "unavailable"
        if not enabled:
            return "disabled"
        return "ran" if score_present else "abstained"

    return {
        "base": "ran" if wa else "not run",
        "semantic": optional(
            bool(_HAS_WATER_DETECTION and water_config.ENABLE_SEMANTIC_CUE),
            _HAS_SEMANTIC_WATER,
            wa.get("semanticScore") is not None,
        ),
        "residual": optional(
            bool(_HAS_WATER_DETECTION and water_config.ENABLE_RESIDUAL_CUE),
            _HAS_INTRINSIC,
            wa.get("residualScore") is not None,
        ),
        "segmenter": optional(
            bool(_HAS_WATER_DETECTION and water_config.ENABLE_WATER_SEGMENTER),
            _HAS_WATER_SEGMENTER,
            wa.get("segmenterCoverage") is not None,
        ),
    }


def _module_status(representative: Optional[Dict]) -> Dict[str, Dict]:
    """
    Report which analysis witnesses actually ran.

    Every optional module is behind an import gate and a broad exception
    handler, so a missing dependency previously disabled a whole phase with no
    warning while the response still looked complete. A confident answer with a
    silently absent witness is worse than a visibly degraded one, so the state
    is now reported explicitly and surfaced in the UI.

    Reported for the REPRESENTATIVE pothole — the same one that supplies the
    headline verdict — so status and verdict always describe one pothole.
    """
    first = representative or {}
    geo = first.get("geometryAnalysis", {}) or {}
    has_dinov2 = bool(geo.get("foundationFeatures"))

    return {
        "geometry": {
            "available": bool(_HAS_GEOMETRY),
            "ran": bool(geo.get("curvatureFeatures")),
            "detail": "curvature, road-relative bowl depth, surface normals",
        },
        "semantic": {
            "available": bool(_HAS_GEOMETRY) and has_dinov2,
            "ran": has_dinov2,
            "calibrated": semantic_config.IS_CALIBRATED,
            "verdict": first.get("semanticVerdict"),
            "detail": semantic_config.describe(),
        },
        "water": {
            "available": bool(_HAS_WATER_DETECTION),
            "ran": first.get("waterAnalysis") is not None,
            "calibrated": water_config.IS_CALIBRATED if _HAS_WATER_DETECTION else False,
            "cues": _water_cue_status(first),
            # Name the rule that actually made the call for this pothole.
            "detail": ("water detection unavailable" if not _HAS_WATER_DETECTION
                       else water_segmenter.describe() + "; cue ensemble: " + water_config.describe()
                       if (first.get("waterAnalysis") or {}).get("combination") == "learned_segmenter"
                       else water_config.describe()),
        },
        "temporal": {
            "available": bool(_HAS_TEMPORAL),
            "ran": first.get("temporalAnalysis") is not None,
            "detail": "age estimation and 30/60/90-day progression",
        },
        "metric": {
            "available": bool(_HAS_METRIC),
            "ran": first.get("metricDepth") is not None,
            "calibrated": bool(_HAS_METRIC),
            "primary": first.get("severitySource") == "metric",
            "detail": (metric_severity.describe() if _HAS_METRIC
                       else "measured-label depth model unavailable (needs MoGe and ml_models/metric/)"),
        },
    }


@app.get("/healthz")
async def healthz():
    return {"success": True, "status": "ok"}


def _parse_classification_report(report_path: Path) -> Optional[Dict[str, object]]:
    """Extract accuracy and macro F1 from sklearn text classification reports."""
    text = report_path.read_text(encoding="utf-8", errors="ignore")

    acc_match = re.search(r"accuracy\s+([0-9]*\.?[0-9]+)", text)
    macro_match = re.search(
        r"macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)",
        text,
    )

    if not acc_match:
        return None

    model_name = report_path.stem.replace("classification_report_", "")
    return {
        "model": model_name,
        "accuracy": float(acc_match.group(1)),
        "macroF1": float(macro_match.group(3)) if macro_match else None,
        "reportFile": report_path.name,
    }


def _load_ablation_rows() -> List[Dict[str, object]]:
    """Load ablation study rows if available."""
    ablation_path = ML_RESULTS_DIR / "ablation_study.csv"
    if not ablation_path.is_file():
        return []

    rows: List[Dict[str, object]] = []
    with ablation_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "subset": row.get("Feature Subset", "Unknown"),
                    "numFeatures": int(float(row.get("Num Features", 0))),
                    "valAccuracy": float(row.get("Val Accuracy", 0.0)),
                    "macroF1": float(row.get("Macro F1", 0.0)),
                }
            )
    return rows


def _load_feature_rows(max_rows: int = 1200) -> Dict[str, object]:
    """Load numeric feature rows for interactive scatter exploration."""
    candidate_paths = [
        ML_RESULTS_DIR / "valid_features.csv",
        ML_RESULTS_DIR / "train_features.csv",
    ]
    data_path = next((p for p in candidate_paths if p.is_file()), None)
    if data_path is None:
        return {"columns": [], "rows": []}

    rows: List[Dict[str, object]] = []
    feature_columns: List[str] = []

    with data_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames:
            feature_columns = [
                c for c in reader.fieldnames if c not in {"img_name", "severity_name", "pothole_idx"}
            ]

        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break

            normalized: Dict[str, object] = {
                "imageName": row.get("img_name", ""),
                "severity": row.get("severity_name", "Unknown"),
            }

            for col in feature_columns:
                value = row.get(col, "")
                try:
                    normalized[col] = float(value)
                except (TypeError, ValueError):
                    normalized[col] = 0.0

            rows.append(normalized)

    return {
        "columns": feature_columns,
        "rows": rows,
    }


def _graph_category(file_name: str) -> str:
    lower = file_name.lower()
    for prefix, category in GRAPH_CATEGORY_MAP:
        if lower.startswith(prefix):
            return category
    return "Other"


def _build_graph_manifest() -> List[Dict[str, str]]:
    """Build graph manifest for all PNG outputs in ml_results."""
    if not ML_RESULTS_DIR.is_dir():
        return []

    graphs = []
    for png_path in sorted(ML_RESULTS_DIR.glob("*.png")):
        display_name = png_path.stem.replace("_", " ").title()
        graphs.append(
            {
                "file": png_path.name,
                "title": display_name,
                "category": _graph_category(png_path.name),
                "url": f"/insights/files/{png_path.name}",
            }
        )
    return graphs


@app.post("/analyze")
def analyze_image(file: UploadFile = File(...)):
    # Deliberately NOT `async def`. Every stage below is synchronous CPU/GPU
    # work; inside a coroutine it would block the event loop for the whole scan,
    # stalling every other request — including /healthz — on a single worker.
    # A plain `def` endpoint is executed in the threadpool instead.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        with _PIPELINE_LOCK:
            return _run_analysis(tmp_path)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _run_analysis(tmp_path: str) -> JSONResponse:
    """The full per-image pipeline. Caller owns the temp file and the lock."""
    # ── 1. Read image and segment all potholes ──
    original_image = cv2.imread(tmp_path)
    if original_image is None:
        raise FileNotFoundError("Unable to read uploaded image")

    masks = get_all_masks(tmp_path)

    # ── 2. Depth estimation ──
    depth_map = get_depth_map(original_image)
    if depth_map.shape != original_image.shape[:2]:
        depth_map = cv2.resize(
            depth_map,
            (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # ── 2b. Image-level water cue inputs ──
    # Computed once per IMAGE, never per pothole: both are full forward
    # passes, and a frame with seven potholes would otherwise pay for them
    # seven times. Both gated off by default — see water_config.
    water_prior = None
    residual_map = None
    if _HAS_WATER_DETECTION:
        if _HAS_SEMANTIC_WATER and water_config.ENABLE_SEMANTIC_CUE:
            try:
                water_prior = semantic_water.water_prior_map(
                    cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                )
            except Exception:
                water_prior = None
        if _HAS_INTRINSIC and water_config.ENABLE_RESIDUAL_CUE:
            try:
                residual_map = intrinsic_cues.residual_energy_map(
                    cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                )
            except Exception:
                residual_map = None

    # Learned water segmenter: one pass per image; None falls back to the ensemble.
    water_seg_map = None
    if _HAS_WATER_DETECTION and _HAS_WATER_SEGMENTER:
        water_seg_map = water_segmenter.water_map(original_image)

    # ── 2c. Measured-label depth model: MoGe once per image ──
    # Measured against RealSense on 742 YOLO-detected potholes, the legacy vote
    # was right 25.5% of the time (below always answering "Moderate"); this
    # model 50.7%, with fewer under-reported hazards (56 against 85).
    metric_geometry = None
    if _HAS_METRIC and masks:
        metric_geometry = metric_severity.geometry_for_image(original_image, masks)
    metric_primary = metric_geometry is not None and metric_severity.is_primary()

    # ── 3. Load ML artifacts once ──
    scaler = None
    ml_models = {}
    try:
        scaler, ml_models = load_ml_artifacts()
    except FileNotFoundError:
        scaler, ml_models = None, {}

    expected_features = int(getattr(scaler, "n_features_in_", 11)) if scaler is not None else 11

    # ── 4. Per-pothole analysis ──
    potholes: List[Dict[str, object]] = []
    mask_records: List[Tuple[np.ndarray, Dict[str, object]]] = []

    for idx, mask in enumerate(masks, start=1):
        depth_features = extract_depth_features(mask, depth_map)
        if depth_features is None:
            continue

        rule_severity = normalize_severity(classify_severity(depth_features))
        ml_predictions: Dict[str, str] = {}

        ml_feature_vec = extract_ml_features(mask, depth_map, expected_features)
        if ml_feature_vec is not None and scaler is not None and ml_models:
            X_scaled = scaler.transform(ml_feature_vec)
            for name, model in ml_models.items():
                label = int(model.predict(X_scaled)[0])
                severity = normalize_severity(SEVERITY_MAP.get(label, f"Unknown({label})"))
                display_name = name.replace("_", " ").title()
                if name == "svm":
                    display_name = "SVM (RBF Kernel)"
                ml_predictions[display_name] = severity

        classifications = {
            "Rule-Based": rule_severity,
            **ml_predictions,
        }

        consensus, consensus_count, total_classifiers = majority_vote(
            list(classifications.values())
        )
        legacy_vote = consensus

        metric = metric_severity.predict(metric_geometry, idx - 1) if metric_geometry is not None else None
        if metric is not None:
            classifications["Depth Model (measured)"] = metric["severity"]
            if metric_primary:
                consensus = metric["severity"]
            consensus_count = sum(1 for v in classifications.values() if v == consensus)
            total_classifiers = len(classifications)

        bbox = mask_to_bbox(mask)
        bbox_obj = None
        if bbox is not None:
            bbox_obj = {
                "x1": int(bbox[0]),
                "y1": int(bbox[1]),
                "x2": int(bbox[2]),
                "y2": int(bbox[3]),
            }

        features = feature_bundle(depth_features)
        pothole = {
            "id": int(idx),
            "bbox": bbox_obj,
            "features": features,
            "classifications": classifications,
            "consensusSeverity": str(consensus),
            "consensusCount": int(consensus_count),
            "totalClassifiers": int(total_classifiers),
            "severitySource": "metric" if (metric_primary and metric is not None) else "legacy_vote",
            "legacyVote": str(legacy_vote),
        }
        if metric is not None:
            pothole["metricDepth"] = metric

        # Geometry & DINOv2 features (novel extension — additive only)
        if _HAS_GEOMETRY:
            try:
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                geo_feats = extract_all_geometry_features(mask, depth_map, image_rgb)
                if geo_feats is not None:
                    pothole["geometryAnalysis"] = {
                        "curvatureFeatures": {
                            k: round(float(v), 6) if isinstance(v, (int, float)) else v
                            for k, v in geo_feats.items() if not k.startswith("dinov2")
                        },
                    }

                    # Extract DINOv2 foundation features specifically into a sub-object
                    dinov2_features = {k: v for k, v in geo_feats.items() if k.startswith("dinov2")}
                    if dinov2_features:
                        ratio = dinov2_features.get(semantic_config.RATIO_KEY)
                        n_patches = int(dinov2_features.get("dinov2_patch_count", 0))

                        pothole["geometryAnalysis"]["foundationFeatures"] = {
                            "dissimilarity": round(dinov2_features.get("dinov2_dissimilarity", 0), 4),
                            "insideVariance": round(dinov2_features.get("dinov2_inside_variance", 0), 4),
                            "outsideVariance": round(dinov2_features.get("dinov2_outside_variance", 0), 4),
                            # Corrected, ring-local, size-matched comparison (A1)
                            "ringVariance": round(dinov2_features.get("dinov2_ring_variance", 0), 4),
                            "varianceRatio": round(ratio, 4) if ratio is not None else None,
                            "patchCount": n_patches,
                            "calibrated": semantic_config.IS_CALIBRATED,
                        }

                        # ── Semantic Verification Override ──
                        # A real crater is materially heterogeneous (rubble, cracks,
                        # internal shadow); a flat illusion is smoother than the road
                        # around it. The ratio compares interior against a local ring
                        # of road patches, sampled to the same patch count.
                        #
                        # Two-way: strong smoothness downgrades an over-reported
                        # verdict; strong heterogeneity raises an under-reported one
                        # (the Phase 1 "dry, shadowless flattening" failure mode,
                        # which the old one-way rule structurally could not act on).
                        #
                        # Abstains when the interior covers too few patches for the
                        # variance to mean anything.
                        final, verdict, illusion = semantic_override(
                            consensus, rule_severity, ratio, n_patches)
                        pothole["semanticVerdict"] = verdict
                        if pothole["severitySource"] == "metric":
                            # Advisory only. Applied to measured-depth verdicts the
                            # uncalibrated override raised under-reported potholes
                            # from 56 to 101 of 742 (served_severity_comparison.json).
                            if final != consensus:
                                pothole["semanticSuggestion"] = final
                        else:
                            pothole["consensusSeverity"] = final
                            if illusion:
                                pothole["illusionWarning"] = True

                        # An override replaces the verdict, so the agreement
                        # figure must describe the NEW verdict. consensusCount was
                        # taken from the vote; left alone, a Deep 5/5 downgraded
                        # to Shallow would report "Shallow · 5/5 agree" when no
                        # classifier said Shallow at all.
                        if pothole["consensusSeverity"] != consensus:
                            pothole["votedSeverity"] = str(consensus)
                            pothole["consensusCount"] = sum(
                                1 for v in classifications.values()
                                if v == pothole["consensusSeverity"]
                            )
            except Exception:
                pass  # graceful degradation

        # Water detection (novel extension — additive only)
        if _HAS_WATER_DETECTION:
            try:
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                curv_feats = extract_curvature_features(mask) if _HAS_GEOMETRY else None
                water_result = _detect_water(
                    image_rgb, mask,
                    depth_map=depth_map,
                    curvature_features=curv_feats,
                    water_prior_map=water_prior,
                    residual_map=residual_map,
                    water_segmentation_map=water_seg_map,
                )
                if water_result is not None:
                    pothole["waterAnalysis"] = {
                        "waterDetected": water_result.get('is_water', False),
                        "waterProbability": water_result.get('water_probability', 0.0),
                        "confidenceLevel": water_result.get('confidence_level', 'low'),
                        "inconsistencyScore": water_result.get('inconsistency_score'),
                        "semanticScore": water_result.get('semantic_score'),
                        "residualScore": water_result.get('residual_score'),
                        "cueWeights": water_result.get('cue_weights'),
                        "correctionsApplied": water_result.get('corrections_applied'),
                        # Which decision rule produced the probability:
                        # "logistic" (calibrated), or a fallback when a
                        # calibrated cue was unavailable.
                        "combination": water_result.get('combination'),
                        # With the learned segmenter deciding, the share of the
                        # pothole it marks as water; the cue ensemble's own
                        # probability is kept for comparison.
                        "segmenterCoverage": water_result.get('segmenter_coverage'),
                        "ensembleProbability": water_result.get('ensemble_probability'),
                    }
            except Exception:
                pass  # graceful degradation

        # Temporal Analysis (novel extension — additive only)
        if _HAS_TEMPORAL:
            try:
                image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                age_estimate = estimate_pothole_age(mask, image_rgb)
                if age_estimate:
                    # Forecast from the verdict the user is shown (consensus,
                    # after any semantic override), not the rule-based vote —
                    # otherwise the timeline starts from one severity and
                    # "progresses" from another.
                    progression = predict_severity_progression(
                        current_severity=pothole["consensusSeverity"],
                        age_estimate=age_estimate,
                        weather_context="freeze_thaw"
                    )
                    pothole["temporalAnalysis"] = {
                        "ageCategory": age_estimate["age_category"],
                        "ageScore": round(age_estimate["age_score"], 2),
                        "ageDescription": age_estimate["age_description"],
                        "edgeSharpness": round(age_estimate["edge_sharpness"], 2),
                        "crackTexture": round(age_estimate["crack_texture_score"], 2),
                        "progression": {
                            "30d": progression["prediction_30d"]["severity"],
                            "60d": progression["prediction_60d"]["severity"],
                            "90d": progression["prediction_90d"]["severity"],
                        }
                    }
            except Exception:
                pass

        # Depth cross-section profile for interpretable charting
        try:
            water_detected = pothole.get("waterAnalysis", {}).get("waterDetected", False)
            severity = pothole.get("consensusSeverity", "Moderate")
            image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

            # If an illusion was detected, physically flatten the depth map inside the mask 
            # before generating the cross-section chart to reflect true shallow geometry.
            chart_depth_map = depth_map
            if pothole.get("illusionWarning"):
                chart_depth_map = depth_map.copy()
                road_mean = np.mean(chart_depth_map[mask == 0]) if np.any(mask == 0) else np.mean(chart_depth_map)
                chart_depth_map = np.where(mask > 0, 
                                           chart_depth_map * 0.15 + (road_mean * 0.85), 
                                           chart_depth_map)

            depth_profile = compute_depth_slice(
                mask, chart_depth_map, water_detected=water_detected, severity=severity, image_gray=image_gray
            )
            if depth_profile:
                pothole["depthProfile"] = depth_profile
        except Exception:
            pass  # graceful degradation

        potholes.append(pothole)
        mask_records.append((mask, pothole))

    # Choose one representative pothole: the most severe, then the largest. It
    # supplies every image-level field, the module status, and (via
    # representativeId) the pothole the dashboard's diagnostic panels show.
    representative = None
    if potholes:
        representative = max(
            potholes,
            key=lambda p: (
                severity_rank(str(p["consensusSeverity"])),
                float(p["features"].get("pothole_area", 0)),
            ),
        )

    # ── 5. Build Visualization Images ──
    img_original = encode_image(original_image)

    overlay_mask = original_image.copy()
    for mask, pothole in mask_records:
        severity = str(pothole["consensusSeverity"])
        color = SEVERITY_COLORS_BGR.get(severity, (255, 255, 255))

        tint = np.zeros_like(original_image)
        tint[:, :, 0] = color[0]
        tint[:, :, 1] = color[1]
        tint[:, :, 2] = color[2]
        blended = cv2.addWeighted(original_image, 0.45, tint, 0.55, 0)
        overlay_mask[mask == 1] = blended[mask == 1]

        bbox_obj = pothole.get("bbox")
        if bbox_obj is not None:
            bbox = (
                int(bbox_obj["x1"]),
                int(bbox_obj["y1"]),
                int(bbox_obj["x2"]),
                int(bbox_obj["y2"]),
            )
            lines = [
                f"P{pothole['id']}: {severity}",
                f"Area: {pothole['features'].get('pothole_area', 0)}",
                f"Drop: {pothole['features'].get('local_depth_contrast', 0.0):.3f}",
            ]
            draw_labeled_bbox(overlay_mask, bbox, lines, box_color=color)

    img_mask = encode_image(overlay_mask)

    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    img_depth = encode_image(depth_heatmap)

    # Annotated depth overlay (contours, markers, slice lines)
    depth_annotated_img = build_depth_annotated_image(
        original_image, depth_map, mask_records,
    )
    img_depth_annotated = encode_image(depth_annotated_img)

    schematic_img = build_schematic_image(original_image.shape, potholes)
    img_schematic = encode_image(schematic_img)

    if representative is None:
        consensus_text = "No Pothole"
        consensus_subtext = "Detected by 1 of 1 classifiers"
        features = feature_bundle(None)
        classifications = {"Rule-Based": "No Pothole"}
        consensus_count = 1
        total_classifiers = 1
        bbox = None
    else:
        consensus_text = str(representative["consensusSeverity"])
        pothole_count = len(potholes)
        consensus_subtext = f"Worst severity among {pothole_count} pothole(s)"
        features = representative["features"]
        classifications = representative["classifications"]
        consensus_count = int(representative["consensusCount"])
        total_classifiers = int(representative["totalClassifiers"])
        bbox = representative["bbox"]

    # Detect if any pothole has water
    has_water_filled = any(
        p.get("waterAnalysis", {}).get("waterDetected", False)
        for p in potholes
    )

    return JSONResponse(
        {
            "success": True,
            "potholeCount": len(potholes),
            "consensusSeverity": consensus_text,
            "consensusSubtext": consensus_subtext,
            "consensusCount": consensus_count,
            "totalClassifiers": total_classifiers,
            "features": features,
            "classifications": classifications,
            "moduleStatus": _module_status(representative),
            "potholes": potholes,
            # The pothole every image-level field above was taken from. The UI
            # uses it for the per-pothole panels too, so the headline verdict and
            # the water / temporal / semantic diagnostics describe the same one.
            "representativeId": representative["id"] if representative else None,
            # Which signal produced consensusSeverity, and the measured-label
            # depth estimate behind it when that was the metric model.
            "severitySource": representative.get("severitySource", "legacy_vote") if representative else None,
            "metricDepth": representative.get("metricDepth") if representative else None,
            "images": {
                "original": img_original,
                "maskOverlay": img_mask,
                "depthHeatmap": img_depth,
                "depthAnnotated": img_depth_annotated,
                "schematic": img_schematic,
            },
            "bbox": bbox,
            "hasWaterFilledPotholes": has_water_filled,
        }
    )


@app.get("/insights/summary")
async def insights_summary():
    """Return structured model-performance data and graph manifest for the UI."""
    if not ML_RESULTS_DIR.is_dir():
        return JSONResponse(
            {
                "success": False,
                "error": "ml_results directory not found",
                "metrics": [],
                "ablation": [],
                "featureColumns": [],
                "featureRows": [],
                "graphs": [],
            },
            status_code=404,
        )

    metrics: List[Dict[str, object]] = []
    for report_path in sorted(ML_RESULTS_DIR.glob("classification_report_*.txt")):
        parsed = _parse_classification_report(report_path)
        if parsed is not None:
            metrics.append(parsed)

    metrics.sort(key=lambda row: row.get("accuracy", 0.0), reverse=True)
    top_model = metrics[0] if metrics else None

    ablation_rows = _load_ablation_rows()
    feature_data = _load_feature_rows(max_rows=1200)
    graphs = _build_graph_manifest()

    return JSONResponse(
        {
            "success": True,
            "topModel": top_model,
            "metrics": metrics,
            "ablation": ablation_rows,
            "featureColumns": feature_data["columns"],
            "featureRows": feature_data["rows"],
            "graphs": graphs,
            "totalGraphs": len(graphs),
        }
    )


@app.get("/insights/files/{file_name}")
async def insights_file(file_name: str):
    """Serve image/text/csv artifacts from ml_results for insights dashboard."""
    allowed_ext = {".png", ".txt", ".csv"}
    suffix = Path(file_name).suffix.lower()
    if suffix not in allowed_ext:
        raise HTTPException(status_code=400, detail="Unsupported file extension")

    target = _safe_results_file(file_name)
    return FileResponse(path=str(target), filename=target.name)

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=reload_enabled)
