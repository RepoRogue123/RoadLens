# pip install reportlab tqdm
import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import cv2
import joblib
from matplotlib import pyplot as plt
import io

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from segmentation import get_all_masks
from features import extract_features_extended
from classifier import classify_severity as rule_based_classify
from inference import get_depth_map, _load_depth_model, extract_ml_features

try:
    from temporal_analysis import estimate_pothole_age, predict_severity_progression
    HAS_TEMPORAL = True
except ImportError:
    HAS_TEMPORAL = False

SEVERITY_MAP = {0: "Shallow", 1: "Moderate", 2: "Deep"}
WEIGHTS = {"Deep": 3, "Moderate": 2, "Shallow": 1, "No pothole": 0, "Unknown": 0}

def load_ml_models():
    base_dir = Path("ml_models/extended")
    if not base_dir.exists():
        base_dir = Path("ml_models")
        
    scaler_path = base_dir / "feature_scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
        
    models = {}
    for pkl in base_dir.glob("*.pkl"):
        if pkl.name == "feature_scaler.pkl":
            continue
        try:
            models[pkl.stem] = joblib.load(pkl)
        except Exception as e:
            print(f"Skipping {pkl.name}: {e}")
            
    return scaler, models

def majority_vote(predictions):
    if not predictions:
        return "Unknown"
    c = Counter(predictions)
    # Get most common, if tied, it will just pick one arbitrarily
    return c.most_common(1)[0][0]

def analyze_segment(images_folder, output_dir):
    print(f"Analyzing road segment: {images_folder}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    scaler, ml_models = load_ml_models()
    if not scaler or not ml_models:
        print("Warning: ML models or scaler not found. Proceeding with rule-based only.")
        
    # Pre-load depth model
    print("Loading depth model...")
    _load_depth_model()
    
    image_paths = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
        image_paths.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
    image_paths = list(set(image_paths))
    
    results = []
    
    total_images_processed = 0
    total_images_with_potholes = 0
    total_potholes_detected = 0
    severity_distribution = {"Shallow": 0, "Moderate": 0, "Deep": 0}
    worst_severity_found = "Shallow"
    total_area_px = 0
    total_max_depth = 0
    total_volume_approx_cm3 = 0
    score_numerator = 0
    
    pixel_size_cm = 0.5
    depth_scale_cm = 50

    # The feature vector must match the scaler's width exactly (11, 20 or 39
    # columns, in training order). inference.extract_ml_features owns that
    # contract; a hand-built 20-column vector here silently failed against the
    # 39-feature scaler and left every report rule-only.
    expected_features = int(getattr(scaler, 'n_features_in_', 11)) if scaler is not None else 11
    ml_failures = 0

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        total_images_processed += 1
        
        try:
            masks = get_all_masks(img_path)
        except Exception as e:
            print(f"Error segmenting {img_name}: {e}")
            continue
            
        if not masks:
            continue
            
        total_images_with_potholes += 1
        
        # Load image for depth map
        image = cv2.imread(img_path)
        depth_map = get_depth_map(image)
        if masks[0].shape != depth_map.shape[:2]:
            depth_map = cv2.resize(depth_map, (masks[0].shape[1], masks[0].shape[0]), interpolation=cv2.INTER_LINEAR)
            
        img_worst_severity = "Shallow"
        
        for i, mask in enumerate(masks):
            features = extract_features_extended(mask, depth_map)
            if not features:
                continue
                
            rule_verdict = rule_based_classify(features)
            
            pothole_area_px = features.get('pothole_area', 0)
            max_depth = features.get('max_depth', 0)
            
            # Use raw depth relative map directly
            depth_inside = depth_map[mask > 0]
            volume_cm3 = float(depth_inside.sum() * (pixel_size_cm**2) * depth_scale_cm)
            
            ml_preds = {}
            if ml_models and scaler:
                feat_array = extract_ml_features(mask, depth_map, expected_features)
                if feat_array is None:
                    ml_failures += 1
                else:
                    X_scaled = scaler.transform(feat_array)
                    for m_name, model in ml_models.items():
                        try:
                            pred = int(model.predict(X_scaled)[0])
                            ml_preds[m_name] = SEVERITY_MAP.get(pred, "Unknown")
                        except Exception as e:
                            # One incompatible pickle must not silence the rest,
                            # and must not fail invisibly either.
                            ml_failures += 1
                            print(f"  !! {m_name} failed on {img_name}#{i}: {type(e).__name__}: {e}")
                    
            all_verdicts = [rule_verdict] + list(ml_preds.values())
            all_verdicts = [v for v in all_verdicts if v in SEVERITY_MAP.values()]
            consensus = majority_vote(all_verdicts) if all_verdicts else rule_verdict
            
            # If still Unknown, default to Rule Based
            if consensus not in SEVERITY_MAP.values():
                consensus = rule_verdict
                
            conf_score = sum(1 for v in all_verdicts if v == consensus) / max(len(all_verdicts), 1)
            
            if WEIGHTS.get(consensus, 0) > WEIGHTS.get(img_worst_severity, 0):
                img_worst_severity = consensus
            if WEIGHTS.get(consensus, 0) > WEIGHTS.get(worst_severity_found, 0):
                worst_severity_found = consensus
                
            # Temporal Analysis Integration
            age_category = "Unknown"
            age_score = 0.5
            age_factor = 1.0
            
            if HAS_TEMPORAL:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                age_estimate = estimate_pothole_age(mask, image_rgb)
                if age_estimate:
                    age_category = age_estimate.get("age_category", "Unknown")
                    age_score = age_estimate.get("age_score", 0.5)
                    
                    # Age affects priority: Chronic (>0.75) gets 1.5x boost, Fresh (<0.25) gets 0.8x
                    age_factor = 0.5 + age_score * 1.33
                    age_factor = max(0.5, min(2.0, age_factor))
                    
                    # Predict progression (optional, just to save in results)
                    progression = predict_severity_progression(consensus, age_estimate)
                    
            if consensus in severity_distribution:
                severity_distribution[consensus] += 1
                
            total_potholes_detected += 1
            total_area_px += pothole_area_px
            total_max_depth += max_depth
            total_volume_approx_cm3 += volume_cm3
            
            # Base score from severity and area
            base_score = WEIGHTS.get(consensus, 0) * pothole_area_px
            # Final score factored by age (older = higher priority)
            score_numerator += base_score * age_factor
            
            row = {
                'image_name': img_name,
                'pothole_id': i,
                'rule_verdict': rule_verdict,
                'consensus_verdict': consensus,
                'confidence_score': round(conf_score, 2),
                'area_px': pothole_area_px,
                'surface_area_cm2': features.get('surface_area_cm2', 0),
                'max_depth': max_depth,
                'mean_depth': features.get('mean_depth', 0),
                'volume_approx_cm3': volume_cm3,
                'age_category': age_category,
                'age_factor': round(age_factor, 2)
            }
            
            # specifically requested fields
            if 'random_forest' in ml_preds:
                row['rf_verdict'] = ml_preds['random_forest']
            if 'xgboost' in ml_preds:
                row['xgb_verdict'] = ml_preds['xgboost']
            if 'lightgbm' in ml_preds:
                row['lgbm_verdict'] = ml_preds['lightgbm']
            if 'ensemble' in ml_preds:
                row['ensemble_verdict'] = ml_preds['ensemble']
                
            results.append(row)
            
    df_results = pd.DataFrame(results)
    if ml_failures:
        print(f"Warning: {ml_failures} ML prediction failure(s); affected potholes used fewer voters.")

    max_possible = 3 * total_area_px
    repair_priority_score = (score_numerator / max_possible + 1e-8) * 100 if max_possible > 0 else 0
    priority_score = min(100.0, max(0.0, repair_priority_score))
    
    volume_ft3 = total_volume_approx_cm3 / 28316.8
    cost_usd = volume_ft3 * 50
    
    summary = {
        "total_images_processed": total_images_processed,
        "total_images_with_potholes": total_images_with_potholes,
        "total_potholes_detected": total_potholes_detected,
        "avg_potholes_per_image": total_potholes_detected / max(total_images_processed, 1),
        "severity_distribution": severity_distribution,
        "worst_severity_found": worst_severity_found,
        "avg_pothole_area_px": total_area_px / max(total_potholes_detected, 1),
        "avg_max_depth": total_max_depth / max(total_potholes_detected, 1),
        "total_volume_approx_cm3": total_volume_approx_cm3,
        "repair_priority_score": round(priority_score, 2),
        "estimated_repair_cost_usd": round(cost_usd, 2)
    }
    
    # Save CSV and JSON
    df_results.to_csv(os.path.join(output_dir, 'segment_results.csv'), index=False)
    with open(os.path.join(output_dir, 'segment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    generate_pdf_report(output_dir, images_folder, summary, df_results)
    print(f"Results saved to {output_dir}")

def generate_pdf_report(output_dir, images_folder, summary, df_results):
    doc = SimpleDocTemplate(os.path.join(output_dir, "road_segment_report.pdf"), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Page 1 Header
    elements.append(Paragraph("Road Segment Pothole Analysis Report", styles['Title']))
    from datetime import datetime
    elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Paragraph(f"Folder analyzed: {images_folder}", styles['Normal']))
    elements.append(Paragraph(f"Total images: {summary['total_images_processed']} | Total potholes: {summary['total_potholes_detected']}", styles['Normal']))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Priority color
    pri_score = summary['repair_priority_score']
    if pri_score <= 30:
        pri_color = colors.green
        pri_text = "LOW PRIORITY"
    elif pri_score <= 60:
        pri_color = colors.orange
        pri_text = "MEDIUM PRIORITY"
    else:
        pri_color = colors.red
        pri_text = "HIGH PRIORITY"
        
    pri_style = ParagraphStyle('Priority', parent=styles['Normal'], textColor=pri_color, fontSize=14, spaceAfter=14)
    elements.append(Paragraph(f"Repair Priority Score: {pri_score} - {pri_text}", pri_style))
    
    # Summary Table
    data = [
        ["Metric", "Value"],
        ["Images Analyzed", str(summary['total_images_processed'])],
        ["Images w/ Potholes", str(summary['total_images_with_potholes'])],
        ["Total Potholes", str(summary['total_potholes_detected'])],
        ["Avg Area (px)", f"{summary['avg_pothole_area_px']:.2f}"],
        ["Avg Max Depth", f"{summary['avg_max_depth']:.4f}"],
        ["Total Approx Volume (cm3)", f"{summary['total_volume_approx_cm3']:.2f}"],
        ["Est. Repair Cost (USD)", f"${summary['estimated_repair_cost_usd']:.2f}"]
    ]
    t = Table(data, colWidths=[3*inch, 2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * inch))
    
    # Page 2 Chart Setup
    plt.figure(figsize=(6, 4))
    sizes = [summary['severity_distribution']['Shallow'], summary['severity_distribution']['Moderate'], summary['severity_distribution']['Deep']]
    sns_colors = ['yellow', 'orange', 'red']
    plt.bar(['Shallow', 'Moderate', 'Deep'], sizes, color=sns_colors)
    plt.title("Severity Distribution")
    imgdata = io.BytesIO()
    plt.savefig(imgdata, format='png')
    imgdata.seek(0)
    plt.close()
    
    elements.append(Image(imgdata, width=5*inch, height=3.5*inch))
    
    # Page 2 Results Table
    elements.append(Paragraph("Per-Image Results Overview", styles['Heading2']))
    if not df_results.empty:
        img_groups = df_results.groupby('image_name')
        img_table = [["Image Name", "Potholes Found", "Worst Severity", "Priority Score"]]
        for name, group in img_groups:
            worst = "Shallow"
            score = 0
            for v in group['consensus_verdict']:
                if WEIGHTS.get(v, 0) > WEIGHTS.get(worst, 0):
                    worst = v
                score += WEIGHTS.get(v, 0)
            img_table.append([name, str(len(group)), worst, str(score)])
            
        img_table.sort(key=lambda x: WEIGHTS.get(x[2], 0), reverse=True)
        t2 = Table(img_table[:20], colWidths=[2.5*inch, 1*inch, 1.5*inch, 1*inch]) # Top 20 max to avoid huge tables
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 8)
        ]))
        elements.append(t2)
        
    elements.append(Spacer(1, 0.5 * inch))
    
    # Page 3 Top 5
    elements.append(Paragraph("Top 5 Most Severe Potholes", styles['Heading2']))
    if not df_results.empty:
        top5 = df_results.sort_values(by=['volume_approx_cm3'], ascending=False).head(5)
        top5_data = [["Image", "ID", "Verdict", "Area", "Depth", "Volume"]]
        for _, r in top5.iterrows():
            top5_data.append([
                r['image_name'], str(r['pothole_id']), r['consensus_verdict'],
                str(int(r['area_px'])), f"{r['max_depth']:.2f}", f"{r['volume_approx_cm3']:.0f}"
            ])
        t3 = Table(top5_data)
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkred),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('FONTSIZE', (0,0), (-1,-1), 8)
        ]))
        elements.append(t3)
        
    elements.append(Spacer(1, 0.5 * inch))
    
    # Disclaimers
    elements.append(Paragraph("IMPORTANT DISCLAIMERS:", styles['Heading3']))
    disc_text = """1. Volume and cost estimates are approximate calculations based on relative depth values, not metric measurements.<br/>
    2. Depth values are relative (0-1 scale) not real-world metric. Pixel size assumed = 0.5cm, depth scale assumed = 50cm max.<br/>
    3. Severity labels are based on pseudo-labels derived from KMeans clustering — not verified ground-truth annotations.<br/>
    4. Professional road inspection is required before any repair decisions are made based on this report."""
    elements.append(Paragraph(disc_text, styles['Normal']))
    
    doc.build(elements)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Segment Analysis")
    parser.add_argument("images_folder", help="Path to folder containing road images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    analyze_segment(args.images_folder, args.output_dir)
