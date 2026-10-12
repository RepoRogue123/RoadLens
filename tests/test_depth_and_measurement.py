"""
Test script for Segmenter v3, Depth Network, and Metric (mm) Measurement.
========================================================================
Place your target image path in IMAGE_PATH below and run via:
    F:\IITJ\GitHub\BTP\btenv\Scripts\python.exe tests/test_depth_and_measurement.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONFIGURE YOUR IMAGE PATH HERE (Change this directly inside the script)
# -------------------------------------------------------------------------
IMAGE_PATH = r"F:/IITJ/BTechProject/crater.jpg"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "ml_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMAGE = OUTPUT_DIR / "depth_measurement_test.png"


def main():
    print("=" * 75)
    print(" RoadLens — Segmenter & Depth Network Verification")
    print("=" * 75)
    print(f"Target Image: {IMAGE_PATH}")

    if not os.path.isfile(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
        sys.exit(1)

    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print(f"ERROR: Failed to load image at {IMAGE_PATH}")
        sys.exit(1)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {w}x{h} px")

    # -------------------------------------------------------------------------
    # 1. Segmenter Verification (Segmenter v3)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[1] SEGMENTER VERIFICATION")
    print("-" * 75)
    import segmentation

    print(f"  Segmenter active model path: {segmentation.MODEL_PATH}")
    print(f"  Confidence threshold:       {segmentation.CONF_THRESHOLD}")

    masks = segmentation.get_all_masks(IMAGE_PATH)
    print(f"  Potholes detected by segmenter: {len(masks)}")

    # -------------------------------------------------------------------------
    # 2. Depth Network & Metric (mm) Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[2] DEPTH NETWORK & METRIC (mm) VERIFICATION")
    print("-" * 75)

    import relief_depth
    import metric_severity

    print(f"  relief_depth.HAS_RELIEF:    {relief_depth.HAS_RELIEF}")
    installed_weights = relief_depth.installed_weights()
    print(f"  Installed relief weights:   {installed_weights if installed_weights else 'NONE found in ml_models/metric/'}")
    print(f"  metric_severity.HAS_METRIC: {metric_severity.HAS_METRIC}")
    print(f"  metric_severity.HAS_REGRESSOR: {metric_severity.HAS_REGRESSOR}")

    metric_relief_map = None
    mm_depths = []

    if relief_depth.HAS_RELIEF:
        print("  --> Running Relief Depth Network for mm-scale depth...")
        metric_relief_map = relief_depth.predict_relief(image_bgr)
        if metric_relief_map is not None:
            print(f"  --> Metric relief map generated: shape {metric_relief_map.shape}, unit = millimetres")
            for idx, mask in enumerate(masks):
                depth_mm = relief_depth.bowl_depth_mm(metric_relief_map, mask)
                mm_depths.append(depth_mm)
                print(f"      Pothole #{idx + 1}: Estimated Bowl Depth = {depth_mm:.1f} mm" if depth_mm is not None else f"      Pothole #{idx + 1}: None")
    else:
        print("  [!] Metric relief network weights (relief_vits*.pth) DO NOT EXIST on disk.")
        print("      (They were listed in .gitignore and never committed to GitHub).")

    # -------------------------------------------------------------------------
    # 3. Baseline Depth-Anything-V2 (Relative Depth)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 75)
    print("[3] BASELINE DEPTH-ANYTHING-V2 (Relative Depth)")
    print("-" * 75)
    import inference

    print("  Generating baseline Depth-Anything-V2 relative depth map...")
    raw_depth = inference.get_depth_map(image_bgr)
    norm_depth = (raw_depth - raw_depth.min()) / (raw_depth.max() - raw_depth.min() + 1e-8)
    print(f"  Raw depth range:  min={raw_depth.min():.4f}, max={raw_depth.max():.4f}")
    print(f"  Normalized range: min={norm_depth.min():.4f}, max={norm_depth.max():.4f} (arbitrary relative scale, NOT mm)")

    # -------------------------------------------------------------------------
    # 4. Visualization & Output
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Segmentation
    overlay = image_rgb.copy()
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        combined_mask = np.maximum(combined_mask, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 50, 50), 3)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            label = f"#{idx+1}"
            if idx < len(mm_depths) and mm_depths[idx] is not None:
                label += f": {mm_depths[idx]:.1f}mm"
            cv2.putText(overlay, label, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    axes[0].imshow(overlay)
    axes[0].set_title(f"Segmenter v3 Output ({len(masks)} detected)")
    axes[0].axis("off")

    # Subplot 2: Depth Map (Metric mm if available, otherwise relative)
    if metric_relief_map is not None:
        im1 = axes[1].imshow(metric_relief_map, cmap="inferno")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Relief Depth (mm below road)")
        axes[1].set_title("Relief Depth Network (True mm)")
    else:
        im1 = axes[1].imshow(norm_depth, cmap="inferno")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Relative Depth (0.0 to 1.0, NO mm)")
        axes[1].set_title("Depth-Anything-V2 (Relative Scale)")
    axes[1].axis("off")

    # Subplot 3: Pothole Region Inspection
    masked_depth = norm_depth.copy()
    if combined_mask.sum() > 0:
        masked_depth[combined_mask == 0] = np.nan
        im2 = axes[2].imshow(masked_depth, cmap="plasma")
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth inside pothole")
        axes[2].set_title("Pothole Interior Depth")
    else:
        axes[2].imshow(image_rgb)
        axes[2].set_title("No pothole mask found")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    plt.close()

    print("\n" + "=" * 75)
    print(f"Visualization saved to: {OUTPUT_IMAGE}")
    print("=" * 75)
    print("SUMMARY OF VERIFICATION:")
    print(f"  - Segmenter:             WORKING (using {Path(segmentation.MODEL_PATH).name})")
    print(f"  - Metric mm Depth Model: {'ACTIVE' if relief_depth.HAS_RELIEF else 'OFFLINE (relief_vits*.pth weights missing)'}")
    print(f"  - Fallback Depth Map:    WORKING (Depth-Anything-V2 relative unitless depth)")
    print("=" * 75)


if __name__ == "__main__":
    main()
