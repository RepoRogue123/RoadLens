"""
Unified Pothole Severity Inference Pipeline
============================================
Takes an image path, runs YOLO segmentation + DepthAnythingV2 depth estimation,
then classifies severity using:
  1. Rule-based manual thresholds  (existing classifier.py)
  2. Saved ML models              (from ml_classifier.py training)

Usage:
    python inference.py <image_path> [--output_dir OUTPUT] [--no_show]
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from classifier import classify_severity          # rule-based
from features import extract_depth_features, extract_features_extended, extract_all_geometry_features, extract_curvature_features
from segmentation import get_all_masks            # YOLO segmentation

# Optional imports for new modules (graceful degradation)
try:
    from water_detection import detect_water as _detect_water
    _HAS_WATER_DETECTION = True
except ImportError:
    _HAS_WATER_DETECTION = False

# Optional SfS integration (Frankot-Chellappa local depth refinement)
try:
    from shape_from_shading import reconstruct_depth_sfs as _reconstruct_sfs
    _HAS_SFS = True
except ImportError:
    _HAS_SFS = False

# When True, replace depth inside pothole mask with SfS reconstruction
# Provides finer local bowl shape than monocular Depth-Anything-V2
USE_SFS_DEPTH = False

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def _resolve_models_dir() -> str:
    """
    Prefer ml_models/extended/ when it actually contains a trained scaler.

    ml_classifier.py writes there when USE_MERGED_DATASET is on, while this
    module historically only ever read ml_models/ — so a retrain silently
    produced models the API never loaded. Preferring the extended directory
    when populated fixes that without disturbing the existing root models.
    """
    extended = os.path.join(SCRIPT_DIR, "ml_models", "extended")
    if os.path.isfile(os.path.join(extended, "feature_scaler.pkl")):
        return extended
    return os.path.join(SCRIPT_DIR, "ml_models")


MODELS_DIR = _resolve_models_dir()

# Feature column orders must match ml_classifier.py training exactly.
FEATURE_COLS_11 = [
    "height", "width", "box_area", "pothole_area", "nonpothole_area",
    "mean_depth", "max_depth", "min_depth", "depth_std", "depth_range",
    "p90_depth",
]

FEATURE_COLS_20 = [
    "height", "width", "box_area", "pothole_area", "nonpothole_area",
    "mean_depth", "max_depth", "min_depth", "depth_std", "depth_range",
    "p90_depth", "aspect_ratio", "solidity", "compactness",
    "depth_skewness", "depth_kurtosis", "boundary_gradient",
    "weighted_mean_depth", "surface_area_px2", "surface_area_cm2",
]

# Phase 1 geometry columns, appended after FEATURE_COLS_20.
# This order MUST match GEOMETRY_FEATURE_COLS in ml_classifier.py exactly —
# the scaler and the models are positional, so a mismatch corrupts every
# prediction silently rather than raising.
GEOMETRY_FEATURE_COLS = [
    "max_curvature", "mean_curvature", "std_curvature", "p90_curvature",
    "high_curvature_fraction", "curvature_entropy", "concave_fraction",
    "curvature_sign_changes", "contour_length", "contour_elongation",
    "mean_bowl_depth", "max_bowl_depth", "std_bowl_depth",
    "mean_road_curvature", "slope_variance",
    "mean_normal_deviation", "max_normal_deviation",
    "std_normal_deviation", "p90_normal_deviation",
]

FEATURE_COLS_39 = FEATURE_COLS_20 + GEOMETRY_FEATURE_COLS

SEVERITY_MAP = {0: "Shallow", 1: "Moderate", 2: "Deep"}

# ML model file names (stem only, stored as .pkl)
ML_MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "svm",
    "naive_bayes",
]

SEVERITY_WEIGHT = {
    "No Pothole": 0,
    "No pothole": 0,
    "Shallow": 1,
    "Moderate": 2,
    "Deep": 3,
}


# ── Depth model (lazy-loaded singleton) ──────────────────────────────────────
_DEPTH_MODEL = None


def _resolve_depth_anything_root() -> str:
    candidates = [
        os.path.join(SCRIPT_DIR, "Depth-Anything-V2"),
        os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Depth-Anything-V2")),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find Depth-Anything-V2. Looked in: " + ", ".join(candidates)
    )


def _load_depth_model():
    global _DEPTH_MODEL
    if _DEPTH_MODEL is not None:
        return _DEPTH_MODEL

    import torch

    depth_root = _resolve_depth_anything_root()
    if depth_root not in sys.path:
        sys.path.append(depth_root)

    # pyrefly: ignore [missing-import]
    from depth_anything_v2.dpt import DepthAnythingV2

    ckpt = os.path.join(depth_root, "checkpoints", "depth_anything_v2_vits.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Depth checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384]
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    _DEPTH_MODEL = model
    return _DEPTH_MODEL


def get_depth_map(image: np.ndarray) -> np.ndarray:
    """Infer an HxW float32 depth map (raw values) from a BGR image."""
    model = _load_depth_model()
    depth = model.infer_image(image).astype(np.float32)
    if depth.shape != image.shape[:2]:
        depth = cv2.resize(
            depth, (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return depth


# ── ML helpers ───────────────────────────────────────────────────────────────
def load_ml_artifacts() -> Tuple[object, Dict[str, object]]:
    """Load scaler + all trained ML models from disk."""
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.pkl")
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f"Feature scaler not found at {scaler_path}. "
            "Run ml_classifier.py first to train models."
        )
    scaler = joblib.load(scaler_path)

    models: Dict[str, object] = {}
    for name in ML_MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.isfile(path):
            models[name] = joblib.load(path)
        else:
            print(f"  ⚠  Model file not found, skipping: {path}")
    return scaler, models


def extract_ml_features(
    mask: np.ndarray, depth_map: np.ndarray, expected_features: int = 11
) -> Optional[np.ndarray]:
    """
    Given a binary pothole mask and a depth map, extract an ML feature vector
    matching the model scaler dimensionality.
    """
    h, w = mask.shape[:2]

    pothole_area = int(np.sum(mask))
    if pothole_area == 0:
        return None

    # Bounding-box geometry
    ys, xs = np.where(mask == 1)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    p_width = max(1, x_max - x_min)
    p_height = max(1, y_max - y_min)
    box_area = p_width * p_height
    nonpothole_area = max(0, box_area - pothole_area)

    # Depth statistics (depth_map is raw; normalize to [0,1] first)
    d = depth_map.astype(np.float32)
    d_min_all, d_max_all = float(d.min()), float(d.max())
    if d_max_all > d_min_all:
        d = (d - d_min_all) / (d_max_all - d_min_all)
    else:
        d = np.zeros_like(d)

    depth_values = d[mask == 1]
    if len(depth_values) == 0:
        return None

    max_depth = float(np.max(depth_values))
    min_depth = float(np.min(depth_values))
    mean_depth = float(np.mean(depth_values))
    depth_std = float(np.std(depth_values))
    depth_range = max_depth - min_depth
    p90_depth = float(np.percentile(depth_values, 90))

    vec11 = np.array([
        p_height, p_width, box_area, pothole_area, nonpothole_area,
        mean_depth, max_depth, min_depth, depth_std, depth_range, p90_depth,
    ], dtype=np.float32)

    if expected_features <= 11:
        return vec11[:expected_features].reshape(1, -1)

    # Extended feature path for models trained with 20 features.
    extended = extract_features_extended(mask, depth_map)
    if extended is None:
        return None

    vec20 = np.array([float(extended.get(k, 0.0)) for k in FEATURE_COLS_20], dtype=np.float32)
    if expected_features == 20:
        return vec20.reshape(1, -1)

    # Models retrained with the Phase 1 geometry block expect 39 features.
    if expected_features == len(FEATURE_COLS_39):
        try:
            from features import extract_all_geometry_features
            geo = extract_all_geometry_features(mask, depth_map) or {}
        except ImportError:
            geo = {}
        combined = {**extended, **geo}
        vec39 = np.array(
            [float(combined.get(k, 0.0)) for k in FEATURE_COLS_39], dtype=np.float32
        )
        return vec39.reshape(1, -1)

    # Fallback for unexpected scaler sizes: use the first N extended features.
    if expected_features < len(FEATURE_COLS_20):
        return vec20[:expected_features].reshape(1, -1)
    return None


def _normalize_severity(value: str) -> str:
    if value == "No pothole":
        return "No Pothole"
    return value


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask == 1)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _majority_vote(verdicts: List[str]) -> Tuple[str, int, int]:
    if not verdicts:
        return "No Pothole", 0, 0

    counts = {v: verdicts.count(v) for v in set(verdicts)}
    ranked = sorted(
        counts.items(),
        key=lambda item: (item[1], SEVERITY_WEIGHT.get(item[0], 0)),
        reverse=True,
    )
    winner = ranked[0][0]
    return winner, counts[winner], len(verdicts)


def _severity_rank(label: str) -> int:
    return SEVERITY_WEIGHT.get(label, 0)


def _serializable_features(depth_features: Optional[Dict[str, float]]) -> Dict[str, float]:
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


# ── main pipeline ────────────────────────────────────────────────────────────
def run_inference(
    image_path: str,
    output_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Full inference pipeline:
      segmentation ➜ depth ➜ features ➜ rule-based + ML predictions.
    """
    print(f"\n{'='*60}")
    print(f"  Pothole Severity Inference Pipeline")
    print(f"  Image: {image_path}")
    print(f"{'='*60}\n")

    # ── 1. Segmentation ─────────────────────────────────────────────────────
    print("[1/4] Running YOLO segmentation...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    masks = get_all_masks(image_path)
    pothole_count = len(masks)
    print(f"  Detected masks: {pothole_count}")

    # ── 2. Depth estimation ──────────────────────────────────────────────────
    print("[2/4] Estimating depth map (DepthAnythingV2)...")
    depth_map = get_depth_map(original_image)
    if depth_map.shape != original_image.shape[:2]:
        depth_map = cv2.resize(
            depth_map, (original_image.shape[1], original_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # ── 2b. Optional SfS depth refinement ────────────────────────────────────
    if USE_SFS_DEPTH and _HAS_SFS:
        print("  Applying SfS depth refinement...")
        # SfS replaces depth inside each mask with Frankot-Chellappa reconstruction
        # This captures fine-grained bowl curvature that DA-V2 smooths over
        for mask_item in masks:
            if np.sum(mask_item) >= 50:
                try:
                    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                    sfs_height = _reconstruct_sfs(gray, mask_item)
                    if sfs_height is not None:
                        # Scale SfS to match depth map range inside mask
                        d_inside = depth_map[mask_item > 0]
                        if len(d_inside) > 0:
                            d_min, d_max = float(d_inside.min()), float(d_inside.max())
                            sfs_inside = sfs_height[mask_item > 0]
                            if np.std(sfs_inside) > 1e-8:
                                sfs_scaled = d_min + (sfs_height * (d_max - d_min))
                                depth_map[mask_item > 0] = sfs_scaled[mask_item > 0]
                except Exception as sfs_err:
                    print(f"  ⚠  SfS failed for mask: {sfs_err}")

    # ── 3. Feature extraction ────────────────────────────────────────────────
    print("[3/4] Extracting features...")

    scaler = None
    ml_models: Dict[str, object] = {}
    try:
        scaler, ml_models = load_ml_artifacts()
    except FileNotFoundError as err:
        print(f"  ⚠  {err}")

    expected_features = 11
    if scaler is not None:
        expected_features = int(getattr(scaler, "n_features_in_", 11))

    potholes: List[Dict[str, object]] = []
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # precompute once
    for idx, mask in enumerate(masks, start=1):
        depth_features = extract_depth_features(mask, depth_map)
        if depth_features is None:
            continue

        rule_severity = _normalize_severity(classify_severity(depth_features))
        ml_feature_vec = extract_ml_features(mask, depth_map, expected_features)

        ml_predictions: Dict[str, str] = {}
        if ml_feature_vec is not None and scaler is not None and ml_models:
            X_scaled = scaler.transform(ml_feature_vec)
            for name, model in ml_models.items():
                label = int(model.predict(X_scaled)[0])
                ml_predictions[name] = SEVERITY_MAP.get(label, f"Unknown({label})")

        verdicts = [rule_severity] + list(ml_predictions.values())
        consensus, consensus_count, total_classifiers = _majority_vote(verdicts)
        bbox = _bbox_from_mask(mask)

        # ── Geometry features (novel extension) ──
        geometry_features = None
        try:
            geometry_features = extract_all_geometry_features(mask, depth_map, image_rgb)
        except Exception as geo_err:
            print(f"  ⚠  Geometry feature extraction failed for pothole #{idx}: {geo_err}")

        # ── Water detection (novel extension) ──
        water_result = None
        if _HAS_WATER_DETECTION:
            try:
                curvature_feats = extract_curvature_features(mask)
                water_result = _detect_water(
                    image_rgb, mask,
                    depth_map=depth_map,
                    curvature_features=curvature_feats,
                )
            except Exception as water_err:
                print(f"  ⚠  Water detection failed for pothole #{idx}: {water_err}")

        pothole_entry = {
            "id": idx,
            "mask": mask,
            "bbox": bbox,
            "rule_severity": rule_severity,
            "ml_predictions": ml_predictions,
            "consensus_severity": consensus,
            "consensus_count": consensus_count,
            "total_classifiers": total_classifiers,
            "features": _serializable_features(depth_features),
        }

        # Add geometry features if available
        if geometry_features is not None:
            pothole_entry["geometry_features"] = {
                k: round(float(v), 6) if isinstance(v, (int, float)) else v
                for k, v in geometry_features.items()
            }

        # Add water detection if available
        if water_result is not None:
            pothole_entry["water_detection"] = water_result

        potholes.append(pothole_entry)

    # ── 4. Classification ────────────────────────────────────────────────────
    print("[4/4] Classifying severity...\n")
    if potholes:
        for p in potholes:
            print(f"  Pothole #{p['id']}:")
            print(f"    Rule-Based  → {p['rule_severity']}")
            if p["ml_predictions"]:
                for model_name, sev in p["ml_predictions"].items():
                    display_name = model_name.replace("_", " ").title()
                    print(f"    {display_name:12s} → {sev}")
            else:
                print("    ML          → skipped")

            feats = p["features"]
            print(
                f"    Consensus   → {p['consensus_severity']} "
                f"({p['consensus_count']}/{p['total_classifiers']})"
            )
            print(
                f"    Metrics     → area={feats['pothole_area']}, "
                f"max_depth={feats['max_depth']:.4f}, "
                f"drop={feats['local_depth_contrast']:.4f}"
            )

            # Print geometry features if available
            if "geometry_features" in p:
                gf = p["geometry_features"]
                print(
                    f"    Geometry    → max_curv={gf.get('max_curvature', 0):.6f}, "
                    f"bowl={gf.get('mean_bowl_depth', 0):.4f}, "
                    f"normal_dev={gf.get('mean_normal_deviation', 0):.2f}°"
                )

            # Print water detection if available
            if "water_detection" in p:
                wd = p["water_detection"]
                status = "WATER" if wd.get('is_water') else "DRY"
                prob = wd.get('water_probability', 0)
                conf = wd.get('confidence_level', 'none')
                print(
                    f"    Water       → {status} (prob={prob:.1%}, conf={conf})"
                )
        print()
    else:
        print("  No potholes detected.\n")

    if potholes:
        worst = max(potholes, key=lambda p: _severity_rank(str(p["consensus_severity"])))
        image_consensus_severity = str(worst["consensus_severity"])
    else:
        image_consensus_severity = "No Pothole"

    # ── Visualization ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original + severity labels
    overlay = original_image.copy()
    cv2.putText(
        overlay,
        f"Potholes: {len(potholes)} | Worst: {image_consensus_severity}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    y_offset = 75
    for p in potholes[:8]:
        cv2.putText(
            overlay,
            f"P{p['id']}: {p['consensus_severity']} ({p['consensus_count']}/{p['total_classifiers']})",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 210, 0),
            2,
            cv2.LINE_AA,
        )
        y_offset += 28

    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original + Severity Predictions")
    axes[0].axis("off")

    # Panel 2: Mask overlaid on original image
    overlay_mask = original_image.copy()
    color_palette = [
        (0, 0, 255),
        (0, 165, 255),
        (0, 255, 255),
        (255, 0, 255),
        (255, 120, 0),
        (128, 255, 0),
    ]

    for p in potholes:
        mask = p["mask"]
        bbox = p["bbox"]
        if bbox is None:
            continue

        color = color_palette[(int(p["id"]) - 1) % len(color_palette)]
        tint_layer = np.zeros_like(original_image)
        tint_layer[:, :, 0] = color[0]
        tint_layer[:, :, 1] = color[1]
        tint_layer[:, :, 2] = color[2]
        blended = cv2.addWeighted(original_image, 0.45, tint_layer, 0.55, 0)
        overlay_mask[mask == 1] = blended[mask == 1]

        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay_mask, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay_mask,
            f"P{p['id']}: {p['consensus_severity']}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    axes[1].imshow(cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Pothole Mask Overlay")
    axes[1].axis("off")

    # Panel 3: Depth heatmap
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    depth_heatmap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    axes[2].imshow(cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Depth Map")
    axes[2].axis("off")

    plt.tight_layout()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{stem}_inference.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved visualization → {out_path}")

        schematic_path = os.path.join(output_dir, f"{stem}_schematic.json")
        schematic = {
            "image": image_path,
            "pothole_count": len(potholes),
            "worst_severity": image_consensus_severity,
            "potholes": [
                {
                    "id": int(p["id"]),
                    "bbox": None
                    if p["bbox"] is None
                    else {
                        "x1": int(p["bbox"][0]),
                        "y1": int(p["bbox"][1]),
                        "x2": int(p["bbox"][2]),
                        "y2": int(p["bbox"][3]),
                    },
                    "rule_based": str(p["rule_severity"]),
                    "ml_predictions": {k: str(v) for k, v in p["ml_predictions"].items()},
                    "consensus": str(p["consensus_severity"]),
                    "consensus_count": int(p["consensus_count"]),
                    "total_classifiers": int(p["total_classifiers"]),
                    "features": p["features"],
                    **({"geometry_features": p["geometry_features"]} if "geometry_features" in p else {}),
                    **({"water_detection": p["water_detection"]} if "water_detection" in p else {}),
                }
                for p in potholes
            ],
        }
        with open(schematic_path, "w", encoding="utf-8") as fp:
            json.dump(schematic, fp, indent=2)
        print(f"  Saved schematic     → {schematic_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pothole severity inference (rule-based + ML)"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to save the visualization (default: don't save)",
    )
    parser.add_argument(
        "--no_show", action="store_true",
        help="Suppress interactive plot window",
    )
    args = parser.parse_args()

    run_inference(args.image_path, output_dir=args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
