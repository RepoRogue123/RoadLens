"""
Batch Water Detection & Segmentation Test Harness.

Takes a folder of images, runs YOLOv8 pothole segmentation, runs the learned
water segmenter & cue ensemble on each detected pothole, and produces:
  1. A side-by-side multi-panel diagnostic image for every input photo.
  2. An aggregated visual summary dashboard (00_WATER_SUMMARY_REPORT.png)
     showing statistics, confidence distributions, and an evaluation table.
  3. A machine-readable summary JSON (water_test_summary.json).

Usage:
    # 1. Using default path (configured in script):
    python scripts/test_water_folder.py

    # 2. Specifying a custom folder:
    python scripts/test_water_folder.py --input-dir "path/to/my/images" --limit 20
"""
import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import segmentation
import water_config
import water_detection
import water_segmenter

try:
    import semantic_water
    _HAS_SEMANTIC = True
except Exception:
    _HAS_SEMANTIC = False

# ==============================================================================
# USER CONFIGURATION: EDIT THIS PATH TO TEST YOUR FOLDER OF IMAGES
# ==============================================================================
DEFAULT_INPUT_DIR = os.path.join(
    PROJECT_DIR, "datasets", "water_detection_test_dataset", "water_test_dataset", "IMG"
)
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "water_folder_test")
# ==============================================================================

# Confidence styling palette
CONFIDENCE_COLORS = {
    "high": "#06b6d4",      # Bright Cyan
    "medium": "#0284c7",    # Sky Blue
    "low": "#eab308",       # Amber / Yellow
    "none": "#64748b",      # Slate / Gray
}

CONFIDENCE_BADGES = {
    "high": "HIGH CONFIDENCE WATER",
    "medium": "MEDIUM CONFIDENCE WATER",
    "low": "LOW CONFIDENCE WATER",
    "none": "DRY / NO WATER DETECTED",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Test water detection across a folder of images.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing input images (.jpg, .png, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where side-by-side panels and reports will be saved",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="YOLO pothole segmentation confidence threshold (default: model default)",
    )
    return parser.parse_args()


def create_panel_visualization(
    image_bgr: np.ndarray,
    pothole_masks: List[np.ndarray],
    water_results: List[Dict[str, Any]],
    water_full_map: Optional[np.ndarray],
    image_name: str,
    output_path: str,
):
    """Generates a 4-panel side-by-side diagnostic visualization for an image."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6.5), dpi=150)
    fig.patch.set_facecolor("#0b0f19")  # Modern dark background

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 1: Original Image + Detected Pothole Boundaries
    # ──────────────────────────────────────────────────────────────────────────
    panel1 = image_rgb.copy()
    for idx, mask in enumerate(pothole_masks, start=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel1, contours, -1, (16, 185, 129), 2)  # Emerald green contour

        ys, xs = np.nonzero(mask)
        if len(xs) > 0:
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            cv2.rectangle(panel1, (x1, y1), (x2, y2), (234, 179, 8), 2)  # Yellow bbox
            cv2.putText(
                panel1,
                f"Pothole #{idx}",
                (x1, max(y1 - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (234, 179, 8),
                2,
                cv2.LINE_AA,
            )

    axes[0].imshow(panel1)
    pothole_count_text = f"{len(pothole_masks)} Pothole(s) Found" if pothole_masks else "No Potholes"
    axes[0].set_title(
        f"1. Pothole Segmentation\n({pothole_count_text})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[0].axis("off")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 2: Pothole & Water Mask Overlay
    # ──────────────────────────────────────────────────────────────────────────
    panel2 = image_rgb.copy().astype(np.float32)
    # Tint pothole interior
    combined_potholes = np.zeros((h, w), dtype=bool)
    combined_water = np.zeros((h, w), dtype=bool)

    for mask in pothole_masks:
        combined_potholes |= mask > 0

    if combined_potholes.any():
        # Tint pothole orange-red base
        panel2[combined_potholes] = panel2[combined_potholes] * 0.6 + np.array([249, 115, 22]) * 0.4

    if water_full_map is not None:
        # Water inside potholes in bright cyan
        water_in_pothole = water_full_map & combined_potholes
        if water_in_pothole.any():
            combined_water |= water_in_pothole
            panel2[water_in_pothole] = panel2[water_in_pothole] * 0.4 + np.array([6, 182, 212]) * 0.6

    panel2 = np.clip(panel2, 0, 255).astype(np.uint8)

    # Draw water boundary contours in electric cyan
    if combined_water.any():
        water_contours, _ = cv2.findContours(
            combined_water.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel2, water_contours, -1, (6, 182, 212), 2)

    axes[1].imshow(panel2)
    axes[1].set_title(
        "2. Water in Pothole Overlay\n(Cyan: Water | Orange: Dry Pothole)",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[1].axis("off")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 3: Learned Water Segmenter Map (Full View)
    # ──────────────────────────────────────────────────────────────────────────
    if water_full_map is not None and water_full_map.any():
        seg_vis = np.zeros((h, w, 3), dtype=np.uint8)
        seg_vis[water_full_map] = [6, 182, 212]
        # Blend with grayscale version of original
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        seg_panel = cv2.addWeighted(gray_3ch, 0.4, seg_vis, 0.6, 0)
    else:
        seg_panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        seg_panel = cv2.cvtColor(seg_panel, cv2.COLOR_GRAY2RGB)

    axes[2].imshow(seg_panel)
    water_px_sum = int(water_full_map.sum()) if water_full_map is not None else 0
    axes[2].set_title(
        f"3. YOLOv8 Water Segmenter Map\n(Full Frame Water Pixels: {water_px_sum:,})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    axes[2].axis("off")

    # ──────────────────────────────────────────────────────────────────────────
    # Panel 4: Water Diagnostic Summary Card
    # ──────────────────────────────────────────────────────────────────────────
    axes[3].set_facecolor("#111827")
    axes[3].axis("off")

    # Select representative pothole (the one with highest water probability)
    rep_idx = 0
    rep_res = None
    if water_results:
        rep_idx = max(
            range(len(water_results)),
            key=lambda i: water_results[i].get("water_probability", 0.0),
        )
        rep_res = water_results[rep_idx]

    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)

    if rep_res is None:
        axes[3].text(
            0.5,
            0.5,
            "No Pothole Detected\nIn This Frame",
            color="#94a3b8",
            fontsize=14,
            ha="center",
            va="center",
            fontweight="bold",
        )
    else:
        conf_level = rep_res.get("confidence_level", "none")
        conf_badge = CONFIDENCE_BADGES.get(conf_level, "UNKNOWN")
        badge_color = CONFIDENCE_COLORS.get(conf_level, "#64748b")
        coverage = rep_res.get("segmenter_coverage", 0.0) or 0.0
        prob = rep_res.get("water_probability", 0.0) or 0.0
        is_water = rep_res.get("is_water", False)

        # Draw Badge Header
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor=badge_color, edgecolor="none", alpha=0.9)
        axes[3].text(
            0.5,
            0.92,
            conf_badge,
            color="#ffffff" if conf_level != "low" else "#000000",
            fontsize=11.5,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=bbox_props,
        )

        # Key Metrics List
        metrics_text = (
            f"Evaluated Pothole   : #{rep_idx + 1} of {len(pothole_masks)}\n"
            f"Pothole Mask Area   : {int(pothole_masks[rep_idx].sum()):,} px\n"
            f"Water Surface Ratio : {coverage:.1%}\n"
            f"Model Probability   : {prob:.1%}\n"
            f"Hazard Status       : {'WATER HAZARD' if is_water else 'DRY ROAD'}\n"
            f"Decision Model      : {rep_res.get('combination', 'learned_segmenter')}"
        )
        axes[3].text(
            0.05,
            0.81,
            metrics_text,
            color="#cbd5e1",
            fontsize=9.5,
            va="top",
            linespacing=1.5,
            fontfamily="monospace",
        )

        axes[3].text(
            0.05,
            0.40,
            "Optical Cue Decomposition:",
            color="#38bdf8",
            fontsize=10,
            fontweight="bold",
        )

        # Cue Breakdown Mini Bar Chart at bottom
        cues = [
            ("Edge Smooth", rep_res.get("edge_density_score", 0.0)),
            ("Gradient", rep_res.get("gradient_score", 0.0)),
            ("Specular", rep_res.get("specular_score", 0.0)),
            ("Blue/Color", rep_res.get("color_score", 0.0)),
            ("Saturation", rep_res.get("saturation_score", 0.0)),
        ]
        y_positions = [0.32, 0.25, 0.18, 0.11, 0.04]
        for (cue_name, val), y_pos in zip(cues, y_positions):
            val = val if val is not None else 0.0
            axes[3].text(0.05, y_pos + 0.012, f"{cue_name:11s}", color="#94a3b8", fontsize=8.5, va="center", fontfamily="monospace")
            # Bar background
            axes[3].barh(y_pos + 0.012, 0.45, left=0.40, height=0.035, color="#1e293b")
            # Bar foreground
            bar_color = "#06b6d4" if val > 0.4 else "#3b82f6"
            axes[3].barh(y_pos + 0.012, max(0.45 * val, 0.005), left=0.40, height=0.035, color=bar_color)
            axes[3].text(0.92, y_pos + 0.012, f"{val:.2f}", color="#f1f5f9", fontsize=8.5, va="center", ha="right")

    axes[3].set_title(
        f"4. Water Diagnostics\n({image_name})",
        color="#f8fafc",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    fig.subplots_adjust(top=0.86, bottom=0.06, left=0.02, right=0.98, wspace=0.12)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def generate_summary_dashboard(
    records: List[Dict[str, Any]],
    output_dir: str,
):
    """Creates the final 00_WATER_SUMMARY_REPORT.png visual executive summary."""
    report_path = os.path.join(output_dir, "00_WATER_SUMMARY_REPORT.png")
    total_images = len(records)
    if total_images == 0:
        print("No records to summarize.")
        return

    total_potholes = sum(r["pothole_count"] for r in records)
    images_with_potholes = sum(1 for r in records if r["pothole_count"] > 0)
    images_with_water = sum(1 for r in records if r["water_detected"])

    conf_counts = {"high": 0, "medium": 0, "low": 0, "none": 0}
    for r in records:
        if r["pothole_count"] > 0:
            conf_counts[r["top_confidence"]] = conf_counts.get(r["top_confidence"], 0) + 1
        else:
            conf_counts["none"] = conf_counts.get("none", 0) + 1

    fig = plt.figure(figsize=(18, 12), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    # Grid layout: Top KPI Cards + Middle Charts + Bottom Table
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 2.2, 4.0], hspace=0.35, wspace=0.25)

    # ──────────────────────────────────────────────────────────────────────────
    # Top KPI Cards
    # ──────────────────────────────────────────────────────────────────────────
    kpis = [
        ("Total Images Tested", f"{total_images}", "#3b82f6"),
        ("Images with Potholes", f"{images_with_potholes} ({images_with_potholes/total_images:.0%})", "#10b981"),
        ("Images with Water", f"{images_with_water} ({images_with_water/max(images_with_potholes, 1):.0%})", "#06b6d4"),
    ]

    for i, (title, val, color) in enumerate(kpis):
        ax_kpi = fig.add_subplot(gs[0, i])
        ax_kpi.set_facecolor("#111827")
        ax_kpi.axis("off")
        ax_kpi.text(0.5, 0.65, val, color=color, fontsize=26, fontweight="bold", ha="center", va="center")
        ax_kpi.text(0.5, 0.25, title, color="#94a3b8", fontsize=12, fontweight="medium", ha="center", va="center")
        # Border box
        for spine in ax_kpi.spines.values():
            spine.set_edgecolor("#1e293b")
            spine.set_linewidth(1.5)

    # ──────────────────────────────────────────────────────────────────────────
    # Middle Charts: Confidence Distribution & Water Coverage Distribution
    # ──────────────────────────────────────────────────────────────────────────
    # Chart 1: Confidence Breakdown Bar
    ax_conf = fig.add_subplot(gs[1, 0:2])
    ax_conf.set_facecolor("#111827")
    labels = ["High (>50%)", "Medium (30-50%)", "Low (10-30%)", "Dry / None (<10%)"]
    counts = [conf_counts["high"], conf_counts["medium"], conf_counts["low"], conf_counts["none"]]
    colors = [CONFIDENCE_COLORS["high"], CONFIDENCE_COLORS["medium"], CONFIDENCE_COLORS["low"], CONFIDENCE_COLORS["none"]]

    bars = ax_conf.bar(labels, counts, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.5)
    ax_conf.set_title("Water Presence Confidence Distribution Across Tested Photos", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax_conf.tick_params(colors="#94a3b8", labelsize=10)
    ax_conf.grid(axis="y", color="#1e293b", linestyle="--", alpha=0.7)
    for b in bars:
        height = b.get_height()
        ax_conf.annotate(f"{height}", xy=(b.get_x() + b.get_width() / 2, height),
                         xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                         color="#f1f5f9", fontweight="bold")

    # Chart 2: Donut Chart of Water vs Dry
    ax_pie = fig.add_subplot(gs[1, 2])
    ax_pie.set_facecolor("#111827")
    pie_labels = ["Water Detected", "Dry Road"]
    pie_counts = [images_with_water, total_images - images_with_water]
    pie_colors = ["#06b6d4", "#334155"]
    wedges, texts, autotexts = ax_pie.pie(
        pie_counts,
        labels=pie_labels,
        autopct="%1.1f%%",
        colors=pie_colors,
        startangle=90,
        textprops=dict(color="#f8fafc", fontweight="bold"),
        wedgeprops=dict(width=0.45, edgecolor="#0b0f19", linewidth=2),
    )
    for at in autotexts:
        at.set_color("#ffffff")
    ax_pie.set_title("Overall Water Ratio", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)

    # ──────────────────────────────────────────────────────────────────────────
    # Bottom Table: Summary of Individual Images
    # ──────────────────────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.set_facecolor("#111827")
    ax_tbl.axis("off")
    ax_tbl.set_title("Detailed Batch Execution Manifest (Showing up to 15 entries)", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)

    # Table columns
    headers = ["Image Filename", "Potholes Found", "Water Detected", "Max Water Coverage", "Ensemble Prob", "Confidence Level"]
    rows = []
    display_records = records[:15]
    for r in display_records:
        water_str = "YES" if r["water_detected"] else "NO"
        cov_str = f"{r['top_coverage']:.1%}" if r["pothole_count"] > 0 else "N/A"
        prob_str = f"{r['top_prob']:.1%}" if r["pothole_count"] > 0 else "N/A"
        conf_str = r["top_confidence"].upper() if r["pothole_count"] > 0 else "NO POTHOLE"
        rows.append([r["image_name"], str(r["pothole_count"]), water_str, cov_str, prob_str, conf_str])

    table = ax_tbl.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.6)

    # Style table headers and cells
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#1e293b")
        if row_idx == 0:
            cell.set_facecolor("#1e293b")
            cell.set_text_props(color="#38bdf8", fontweight="bold")
        else:
            rec = display_records[row_idx - 1]
            if rec["water_detected"]:
                cell.set_facecolor("#082f49" if row_idx % 2 == 0 else "#0c4a6e")
                cell.set_text_props(color="#e0f2fe")
            else:
                cell.set_facecolor("#0f172a" if row_idx % 2 == 0 else "#1e293b")
                cell.set_text_props(color="#94a3b8")

    plt.savefig(report_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"\n[OK] Summary Dashboard generated: {report_path}")


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("ROADLENS WATER DETECTION BATCH TEST HARNESS")
    print(f"  Input Directory  : {input_dir}")
    print(f"  Output Directory : {output_dir}")
    print(f"  Limit            : {args.limit or 'All images'}")
    print(f"  Water Segmenter  : {water_segmenter.describe()}")
    print("=" * 80)

    if not os.path.isdir(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Collect valid images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if not image_paths:
        print(f"No image files found in {input_dir}")
        sys.exit(1)

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Processing {len(image_paths)} image(s)...")

    results_summary = []

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        panel_output_path = os.path.join(output_dir, f"{stem}_water_panel.png")

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[{idx}/{len(image_paths)}] Skipping unreadable image: {filename}")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1. Pothole Segmentation
        masks = segmentation.get_all_masks(img_path, conf_threshold=args.conf)

        # 2. Water Segmentation (full-frame)
        water_full_map = water_segmenter.water_map(image_bgr)

        # 3. Semantic Water Prior (CLIPSeg, if active)
        prior_map = None
        if _HAS_SEMANTIC and water_config.ENABLE_SEMANTIC_CUE:
            try:
                prior_map = semantic_water.water_prior_map(image_rgb)
            except Exception:
                prior_map = None

        # 4. Evaluate each detected pothole
        pothole_water_results = []
        for p_idx, mask in enumerate(masks):
            w_res = water_detection.detect_water(
                image_rgb,
                mask,
                water_segmentation_map=water_full_map,
                water_prior_map=prior_map,
            )
            pothole_water_results.append(w_res)

        # 5. Extract representative statistics for the image
        has_water = any(w.get("is_water", False) for w in pothole_water_results)
        max_coverage = max([w.get("segmenter_coverage", 0.0) or 0.0 for w in pothole_water_results], default=0.0)
        max_prob = max([w.get("water_probability", 0.0) or 0.0 for w in pothole_water_results], default=0.0)

        # Representative confidence
        top_conf = "none"
        if pothole_water_results:
            top_pothole = max(pothole_water_results, key=lambda w: w.get("water_probability", 0.0))
            top_conf = top_pothole.get("confidence_level", "none")

        # 6. Render individual 4-panel visual image
        create_panel_visualization(
            image_bgr=image_bgr,
            pothole_masks=masks,
            water_results=pothole_water_results,
            water_full_map=water_full_map,
            image_name=filename,
            output_path=panel_output_path,
        )

        record = {
            "image_name": filename,
            "pothole_count": len(masks),
            "water_detected": has_water,
            "top_coverage": max_coverage,
            "top_prob": max_prob,
            "top_confidence": top_conf,
            "panel_path": panel_output_path,
        }
        results_summary.append(record)

        water_tag = f"WATER ({top_conf.upper()} - {max_coverage:.1%})" if has_water else "DRY"
        print(f"[{idx:3d}/{len(image_paths):3d}] {filename:25s} -> {len(masks)} pothole(s), Status: {water_tag}")

    # 7. Generate aggregate visual summary report
    generate_summary_dashboard(results_summary, output_dir)

    # 8. Save structured JSON summary
    json_path = os.path.join(output_dir, "water_test_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"[OK] Structured JSON manifest saved: {json_path}")
    print("\nBatch testing complete! Output panels saved in:\n ", output_dir)


if __name__ == "__main__":
    main()

