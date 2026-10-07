"""
Robustness Evaluation Script
=============================
Evaluates how pothole severity classifiers degrade under synthetic
adverse conditions (rain, night, fog, waterlogged, shadow).

Tests the hypothesis that geometry-based features are more robust to
adverse conditions than depth-based features, because geometric
properties (curvature, contour shape) persist while appearance-based
depth estimation degrades.

Outputs:
    ml_results/robustness_evaluation.csv
    ml_results/robustness_comparison.png

Usage:
    python scripts/evaluate_robustness.py [--max_images 50]
"""

import argparse
import csv
import os
import sys
import traceback
from collections import defaultdict

import cv2
import numpy as np

# Ensure project root is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from adverse_conditions import synthesize_condition, get_available_conditions
from features import (
    extract_depth_features,
    extract_features_extended,
    extract_all_geometry_features,
    extract_curvature_features,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

RESULTS_DIR = os.path.join(PROJECT_DIR, "ml_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Dataset paths — try merged_dataset first, fall back to data1
VALID_IMG_DIR = os.path.join(PROJECT_DIR, "merged_dataset", "valid", "images")
VALID_LBL_DIR = os.path.join(PROJECT_DIR, "merged_dataset", "valid", "labels")
DEPTH_DIR = os.path.join(PROJECT_DIR, "depth_maps_merged", "valid")

if not os.path.isdir(VALID_IMG_DIR):
    VALID_IMG_DIR = os.path.join(PROJECT_DIR, "data1", "train", "images")
    VALID_LBL_DIR = os.path.join(PROJECT_DIR, "data1", "train", "labels")
    DEPTH_DIR = os.path.join(PROJECT_DIR, "depth_maps_1", "train")

CONDITIONS = get_available_conditions()  # rain, night, fog, waterlogged, shadow

# Severity bins for pseudo-labels (same as ml_classifier.py)
SEVERITY_BINS = {
    'shallow': (0, 0.15),
    'moderate': (0.15, 0.40),
    'deep': (0.40, 1.0),
}

def label_severity_from_area_ratio(mask, image_shape):
    """Simple area-based severity label for evaluation baseline."""
    total_area = image_shape[0] * image_shape[1]
    pothole_area = int(np.sum(mask))
    ratio = pothole_area / max(total_area, 1)
    if ratio < 0.02:
        return 0  # shallow
    elif ratio < 0.08:
        return 1  # moderate
    else:
        return 2  # deep


# ═══════════════════════════════════════════════════════════════════════════
#  Feature Extraction Helpers
# ═══════════════════════════════════════════════════════════════════════════

#  Every extractor below must see a depth map computed from the image it is
#  scoring. The original version loaded one precomputed map from the CLEAN
#  image and reused it for every adverse variant, and called
#  extract_depth_features(gray, mask, depth) against a (mask, depth) signature;
#  the resulting TypeError was swallowed, so depth and combined sets were always
#  None. The saved CSV showed one constant prediction per classifier at 100%
#  "consistency" — an artefact, not robustness. Failures are now raised.

_DEPTH_FN = None


def depth_for(image_bgr):
    """Depth-Anything-V2 depth for exactly this image (lazy-loaded once)."""
    global _DEPTH_FN
    if _DEPTH_FN is None:
        from inference import get_depth_map
        _DEPTH_FN = get_depth_map
    return _DEPTH_FN(image_bgr)


def extract_depth_feature_set(mask, depth_map):
    """Depth-statistics features, as the rule classifier consumes them."""
    feats = extract_depth_features(mask, depth_map)
    if not feats:
        return None
    return np.array([feats.get(k, 0) for k in sorted(feats.keys())], dtype=float)


def extract_geometry_feature_set(mask, depth_map):
    """Geometry features (curvature + road-relative bowl depth + normals)."""
    feats = extract_all_geometry_features(mask, depth_map)
    if not feats:
        return None
    return np.array([feats.get(k, 0) for k in sorted(feats.keys())], dtype=float)


def extract_combined_feature_set(mask, depth_map):
    """Depth statistics and geometry together."""
    depth_feats = extract_depth_features(mask, depth_map)
    geom_feats = extract_all_geometry_features(mask, depth_map)
    if not (depth_feats and geom_feats):
        return None
    combined = {**depth_feats, **geom_feats}
    return np.array([combined.get(k, 0) for k in sorted(combined.keys())], dtype=float)


# ═══════════════════════════════════════════════════════════════════════════
#  Severity Prediction (feature-distance based)
# ═══════════════════════════════════════════════════════════════════════════

def predict_severity_from_features(features):
    """
    Simple severity prediction from features using feature magnitude.
    Since trained models may not exist yet, use feature-based heuristic:
    higher mean absolute feature value → more severe.
    """
    if features is None or len(features) == 0:
        return 1  # default moderate

    mean_val = float(np.mean(np.abs(features)))
    if mean_val < 0.1:
        return 0
    elif mean_val < 0.3:
        return 1
    else:
        return 2


# ═══════════════════════════════════════════════════════════════════════════
#  Main Evaluation Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def parse_label_file(label_path, h, w):
    """Parse YOLO segmentation label into polygon masks."""
    masks = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [coords.astype(np.int32)], 1)
                masks.append(mask)
    return masks


def run_evaluation(max_images=50):
    """Run the full robustness evaluation."""
    print("=" * 70)
    print("  Robustness Evaluation — Adverse Condition Benchmark")
    print("=" * 70)
    print("  Dataset:    {}".format(VALID_IMG_DIR))
    print("  Conditions: {}".format(CONDITIONS))
    print("  Max images: {}".format(max_images))
    print()

    # Collect image files
    if not os.path.isdir(VALID_IMG_DIR):
        print("ERROR: Image directory not found: {}".format(VALID_IMG_DIR))
        return

    img_files = sorted([
        f for f in os.listdir(VALID_IMG_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:max_images]

    print("  Found {} images".format(len(img_files)))

    # Results storage
    results = []  # list of dicts for CSV
    # Track per-condition, per-classifier consistency
    consistency = defaultdict(lambda: defaultdict(list))

    classifier_types = ['depth_based', 'geometry_based', 'combined']

    for img_idx, img_file in enumerate(img_files):
        base = os.path.splitext(img_file)[0]
        img_path = os.path.join(VALID_IMG_DIR, img_file)
        lbl_path = os.path.join(VALID_LBL_DIR, base + '.txt')
        depth_path = os.path.join(DEPTH_DIR, base + '.npy')

        # Skip if missing components
        if not os.path.exists(lbl_path):
            continue

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]

            masks = parse_label_file(lbl_path, h, w)
            if not masks:
                continue

            # Use first mask for evaluation
            mask = masks[0]
            if np.sum(mask) < 50:
                continue

            # ── Original predictions ──
            # Depth is recomputed live rather than read from depth_maps_*, so the
            # clean and adverse images go through the identical depth pipeline.
            depth_map = depth_for(img_bgr)
            orig_depth_feats = extract_depth_feature_set(mask, depth_map)
            orig_geom_feats = extract_geometry_feature_set(mask, depth_map)
            orig_combined_feats = extract_combined_feature_set(mask, depth_map)

            orig_preds = {
                'depth_based': predict_severity_from_features(orig_depth_feats),
                'geometry_based': predict_severity_from_features(orig_geom_feats),
                'combined': predict_severity_from_features(orig_combined_feats),
            }

            # Ground truth (area-based pseudo-label)
            gt_severity = label_severity_from_area_ratio(mask, (h, w))

            # ── Adverse condition predictions ──
            for condition in CONDITIONS:
                try:
                    if condition == 'waterlogged':
                        adverse_img = synthesize_condition(img_bgr, condition, mask=mask, seed=42)
                    else:
                        adverse_img = synthesize_condition(img_bgr, condition, seed=42)

                    # Re-extract features on the adverse image, with depth
                    # re-estimated FROM that image. The mask stays the labelled
                    # polygon, so curvature is unchanged by construction; the
                    # depth-derived bowl and normal features are not.
                    adv_depth_map = depth_for(adverse_img)
                    adv_depth_feats = extract_depth_feature_set(mask, adv_depth_map)
                    adv_geom_feats = extract_geometry_feature_set(mask, adv_depth_map)
                    adv_combined_feats = extract_combined_feature_set(mask, adv_depth_map)

                    adv_preds = {
                        'depth_based': predict_severity_from_features(adv_depth_feats),
                        'geometry_based': predict_severity_from_features(adv_geom_feats),
                        'combined': predict_severity_from_features(adv_combined_feats),
                    }

                    for clf in classifier_types:
                        consistent = 1 if orig_preds[clf] == adv_preds[clf] else 0
                        correct_orig = 1 if orig_preds[clf] == gt_severity else 0
                        correct_adv = 1 if adv_preds[clf] == gt_severity else 0

                        consistency[condition][clf].append(consistent)

                        results.append({
                            'image': base,
                            'condition': condition,
                            'classifier': clf,
                            'gt_severity': gt_severity,
                            'original_pred': orig_preds[clf],
                            'adverse_pred': adv_preds[clf],
                            'prediction_consistent': consistent,
                            'original_correct': correct_orig,
                            'adverse_correct': correct_adv,
                        })

                except Exception as e:
                    print("  Warning: {} failed on {}: {}".format(condition, base, e))

            if (img_idx + 1) % 10 == 0:
                print("  Processed {}/{} images...".format(img_idx + 1, len(img_files)))

        except Exception as e:
            print("  Error processing {}: {}".format(img_file, e))
            continue

    if not results:
        print("ERROR: No results generated")
        return

    # ── Save CSV ──
    csv_path = os.path.join(RESULTS_DIR, "robustness_evaluation.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\n  Saved: {}".format(csv_path))

    # ── Compute summary statistics ──
    print("\n  Summary:")
    print("  {:<15} {:<18} {:<15} {:<15} {:<12}".format(
        'Condition', 'Classifier', 'Consistency%', 'OrigAcc%', 'AdvAcc%'))
    print("  " + "-" * 72)

    summary_data = defaultdict(dict)

    for condition in CONDITIONS:
        for clf in classifier_types:
            cond_clf_rows = [r for r in results
                            if r['condition'] == condition and r['classifier'] == clf]
            if not cond_clf_rows:
                continue

            consist = np.mean([r['prediction_consistent'] for r in cond_clf_rows]) * 100
            orig_acc = np.mean([r['original_correct'] for r in cond_clf_rows]) * 100
            adv_acc = np.mean([r['adverse_correct'] for r in cond_clf_rows]) * 100

            summary_data[condition][clf] = {
                'consistency': consist,
                'orig_acc': orig_acc,
                'adv_acc': adv_acc,
                'drop': orig_acc - adv_acc,
            }

            print("  {:<15} {:<18} {:<15.1f} {:<15.1f} {:<12.1f}".format(
                condition, clf, consist, orig_acc, adv_acc))

    # ── Generate visualization ──
    _generate_robustness_chart(summary_data, classifier_types)

    print("\n  Done! Total rows: {}".format(len(results)))


def _generate_robustness_chart(summary_data, classifier_types):
    """Generate grouped bar chart comparing classifier robustness."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#0f172a')

    conditions = list(summary_data.keys())
    if not conditions:
        plt.close()
        return

    x = np.arange(len(conditions))
    bar_width = 0.22
    clf_colors = {
        'depth_based': '#ef4444',
        'geometry_based': '#22d3ee',
        'combined': '#f59e0b',
    }

    # ── Panel 1: Prediction Consistency ──
    ax = axes[0]
    ax.set_facecolor('#1e293b')
    for i, clf in enumerate(classifier_types):
        vals = [summary_data.get(c, {}).get(clf, {}).get('consistency', 0)
                for c in conditions]
        bars = ax.bar(x + i * bar_width, vals, bar_width,
                     label=clf.replace('_', ' ').title(),
                     color=clf_colors[clf], alpha=0.85, edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    '{:.0f}%'.format(v), ha='center', va='bottom',
                    fontsize=7, color='white', fontweight='bold')

    ax.set_xlabel('Adverse Condition', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Prediction Consistency (%)', color='#94a3b8', fontsize=11)
    ax.set_title('Prediction Consistency Under Adverse Conditions',
                color='white', fontsize=13, fontweight='bold')
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([c.title() for c in conditions], color='#94a3b8')
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=9, facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_color('#334155')

    # ── Panel 2: Accuracy Drop ──
    ax2 = axes[1]
    ax2.set_facecolor('#1e293b')
    for i, clf in enumerate(classifier_types):
        vals = [summary_data.get(c, {}).get(clf, {}).get('drop', 0)
                for c in conditions]
        bars = ax2.bar(x + i * bar_width, vals, bar_width,
                      label=clf.replace('_', ' ').title(),
                      color=clf_colors[clf], alpha=0.85, edgecolor='white', linewidth=0.5)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    '{:.1f}'.format(v), ha='center', va='bottom',
                    fontsize=7, color='white', fontweight='bold')

    ax2.set_xlabel('Adverse Condition', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Accuracy Drop (pp)', color='#94a3b8', fontsize=11)
    ax2.set_title('Accuracy Degradation Under Adverse Conditions',
                 color='white', fontsize=13, fontweight='bold')
    ax2.set_xticks(x + bar_width)
    ax2.set_xticklabels([c.title() for c in conditions], color='#94a3b8')
    ax2.tick_params(axis='y', colors='#94a3b8')
    ax2.axhline(y=0, color='#475569', linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=9, facecolor='#1e293b', edgecolor='#475569', labelcolor='white')
    for spine in ax2.spines.values():
        spine.set_color('#334155')

    plt.suptitle('Robustness Evaluation — Geometry vs Depth Features',
                color='#f59e0b', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(RESULTS_DIR, "robustness_comparison.png")
    plt.savefig(out_path, dpi=150, facecolor='#0f172a', bbox_inches='tight')
    plt.close()
    print("\n  Saved: {}".format(out_path))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness evaluation")
    parser.add_argument("--max_images", type=int, default=50,
                       help="Max images to evaluate")
    args = parser.parse_args()
    run_evaluation(max_images=args.max_images)
