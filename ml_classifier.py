import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import argparse
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
import joblib

# pip install xgboost lightgbm shap umap-learn reportlab
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
import shap

# ── Configuration ──────────────────────────────────────
USE_MERGED_DATASET    = True   # False = use original data1/ only
USE_EXTENDED_FEATURES = True   # False = use original 11 features
USE_REAL_LABELS       = True   # False = KMeans pseudo-labels only
USE_ADVERSE_AUGMENTATION = False # True = Synthetically augment training data with weather/lighting
USE_GEOMETRY_FEATURES = True   # False = depth statistics only (the pre-Phase-1 baseline)

# Phase 1 depth-independent geometry, produced by
# features.extract_all_geometry_features(). Curvature is computed from the 2D
# mask alone; the bowl-depth and surface-normal groups use depth only as a
# difference against the surrounding road. All three groups zero-fill on
# failure, so the column set is stable.
#
# DINOv2 features are deliberately excluded: extract_all_geometry_features is
# called here without image_rgb, and running a ViT forward pass over every
# pothole row would dominate training time. They remain an API-layer signal.
GEOMETRY_FEATURE_COLS = [
    # curvature (mask only — no depth read at all)
    'max_curvature', 'mean_curvature', 'std_curvature', 'p90_curvature',
    'high_curvature_fraction', 'curvature_entropy', 'concave_fraction',
    'curvature_sign_changes', 'contour_length', 'contour_elongation',
    # road-relative bowl depth
    'mean_bowl_depth', 'max_bowl_depth', 'std_bowl_depth',
    'mean_road_curvature', 'slope_variance',
    # surface-normal deviation at the boundary ring
    'mean_normal_deviation', 'max_normal_deviation',
    'std_normal_deviation', 'p90_normal_deviation',
]

if USE_MERGED_DATASET:
    IMAGES_TRAIN = "merged_dataset/train/images"
    LABELS_TRAIN = "merged_dataset/train/labels"
    IMAGES_VALID = "merged_dataset/valid/images"
    LABELS_VALID = "merged_dataset/valid/labels"
    DEPTH_DIR    = "depth_maps_global"
    MODELS_DIR   = "ml_models/extended"
else:
    IMAGES_TRAIN = "data1/train/images"
    LABELS_TRAIN = "data1/train/labels"
    IMAGES_VALID = "data1/valid/images"
    LABELS_VALID = "data1/valid/labels"
    DEPTH_DIR    = "depth_maps_1"
    MODELS_DIR   = "ml_models"

SEVERITY_LABELS_CSV = "merged_dataset/severity_labels/pothole600_annotations.csv"
RANDOM_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# For backwards compatibility with existing code where appropriate
DATA1_PATH = SCRIPT_DIR
DEPTH1_PATH = SCRIPT_DIR
RESULTS_PATH = os.path.join(SCRIPT_DIR, "ml_results")
MODELS_PATH = os.path.join(SCRIPT_DIR, MODELS_DIR)

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)


def parse_yolo_label(label_path: str, img_shape: Tuple[int, int]) -> List[np.ndarray]:
    """Parse YOLO segmentation label and return a list of integer polygon coordinates."""
    h, w = img_shape
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                # Class mapping assumed to be parts[0], followed by polygon points
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                # Scale from normalized [0, 1] to image dimensions
                coords[:, 0] *= w
                coords[:, 1] *= h
                polygons.append(coords.astype(np.int32))
    return polygons


def extract_dataset_features(split: str) -> pd.DataFrame:
    """Extract features from images, labels, and depth maps in a given split."""
    if USE_MERGED_DATASET:
        base_images = IMAGES_TRAIN if split == "train" else IMAGES_VALID
        base_labels = LABELS_TRAIN if split == "train" else LABELS_VALID
        base_depths = os.path.join(DEPTH_DIR, split)
        images_dir = os.path.join(SCRIPT_DIR, base_images)
        labels_dir = os.path.join(SCRIPT_DIR, base_labels)
        depths_dir = os.path.join(SCRIPT_DIR, base_depths)
    else:
        images_dir = os.path.join(DATA1_PATH, split, "images")
        labels_dir = os.path.join(DATA1_PATH, split, "labels")
        depths_dir = os.path.join(DEPTH1_PATH, split)
    
    features_list = []
    
    if not os.path.exists(images_dir):
        print(f"Warning: {images_dir} does not exist. Skipping.")
        return pd.DataFrame()
        
    image_files = os.listdir(images_dir)
    print(f"Extracting features for {split} split...")
    
    for img_name in tqdm(image_files):
        img_basename = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)
        
        # Determine exact label and depth paths
        # Assuming YOLO segmentation label has .txt extension
        label_path = os.path.join(labels_dir, f"{img_basename}.txt")
        # Assuming depth maps have .npy extension as per generate_depth_data1.py
        depth_path = os.path.join(depths_dir, f"{img_basename}.npy")
        
        if not os.path.exists(depth_path):
            # Fallback for slight naming variations
            for filename in os.listdir(depths_dir):
                if filename.startswith(img_basename.split('.rf.')[0]):
                    depth_path = os.path.join(depths_dir, filename)
                    break

        if not os.path.exists(label_path) or not os.path.exists(depth_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        polygons = parse_yolo_label(label_path, (h, w))
        
        try:
            depth_map = np.load(depth_path)
        except Exception as e:
            print(f"Error loading {depth_path}: {e}")
            continue
            
        # Ensure depth map shape matches image
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            
        # For each pothole in the image
        for i, poly in enumerate(polygons):
            # Create binary mask for this pothole
            mask = np.zeros((h, w), dtype=np.uint8)
            # YOLO labels can sometimes be invalid
            if poly.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [poly], 1)
            
            # Extract basic geometry features
            # Bounding box
            x_min, y_min = np.min(poly, axis=0)
            x_max, y_max = np.max(poly, axis=0)
            p_width = max(1, x_max - x_min)
            p_height = max(1, y_max - y_min)
            box_area = p_width * p_height
            
            pothole_area = np.sum(mask)
            nonpothole_area = max(0, box_area - pothole_area) # Prevent negative values technically
            
            if USE_EXTENDED_FEATURES:
                from features import extract_features_extended
                feats = extract_features_extended(mask, depth_map)
            else:
                from features import extract_features
                feats = extract_features(mask, depth_map)
                
            if feats is None:
                continue

            # Extract geometry features (curvature, depth profiles, surface normals)
            try:
                from features import extract_all_geometry_features
                geo_feats = extract_all_geometry_features(mask, depth_map)
                if geo_feats is not None:
                    feats.update(geo_feats)
            except ImportError:
                pass  # geometry features module not available
            except Exception:
                pass  # graceful degradation on individual sample failure
                
            feat_dict = {
                "img_name": img_name,
                "pothole_idx": i
            }
            feat_dict.update(feats)
            features_list.append(feat_dict)
            
    return pd.DataFrame(features_list)

def load_real_labels(features_df, labels_csv_path):
    """
    Loads Pothole-600 severity labels from CSV.
    CSV must have columns: image_name, severity_int, severity_label
    severity_int: 0=Shallow, 1=Moderate, 2=Deep
    """
    if not os.path.exists(labels_csv_path):
        print(f"Warning: Labels CSV not found at {labels_csv_path}. Falling back to pseudo-labels.")
        return features_df['severity_label'].values, ['pseudo'] * len(features_df)
        
    real_labels_df = pd.read_csv(labels_csv_path)
    
    final_labels = []
    label_sources = []
    
    real_count = 0
    pseudo_count = 0
    
    for _, row in features_df.iterrows():
        img_name = row['img_name']
        match = real_labels_df[real_labels_df['image_name'] == img_name]
        if not match.empty:
            final_labels.append(int(match.iloc[0]['severity_int']))
            label_sources.append('real')
            real_count += 1
        else:
            final_labels.append(row['severity_label'])
            label_sources.append('pseudo')
            pseudo_count += 1
            
    print(f"Real labels used: {real_count} samples")
    print(f"Pseudo labels used: {pseudo_count} samples")
    print(f"Total: {len(features_df)} samples")
    
    return np.array(final_labels), label_sources

def generate_improved_pseudolabels(X_scaled_df, feature_names):
    """
    Runs 4 clustering approaches and compares them via silhouette score.
    Returns labels from the method with highest silhouette score.
    """
    print("\nRunning improved clustering comparisons...")
    
    metrics = []
    results_labels = {}
    
    # Method 1 - KMeans k=3 on 2 features
    k2f = KMeans(n_clusters=3, random_state=42, n_init=10)
    if 'max_depth' in X_scaled_df.columns and 'pothole_area' in X_scaled_df.columns:
        X_2f = X_scaled_df[['max_depth', 'pothole_area']]
        labels_1 = k2f.fit_predict(X_2f)
        centroids_1 = k2f.cluster_centers_
        cluster_scores_1 = [(cid, np.linalg.norm(cen)) for cid, cen in enumerate(centroids_1)]
        sorted_clusters_1 = sorted(cluster_scores_1, key=lambda x: x[1])
        mapping_1 = {cid: rank for rank, (cid, _) in enumerate(sorted_clusters_1)}
        labels_1_mapped = np.array([mapping_1[c] for c in labels_1])
        sil_1 = silhouette_score(X_2f, labels_1_mapped)
        dist_1 = {i: np.mean(labels_1_mapped == i) for i in range(3)}
        metrics.append(('KMeans-2feat (orig)', sil_1, dist_1[0]*100, dist_1[1]*100, dist_1[2]*100))
        results_labels['KMeans-2feat (orig)'] = labels_1_mapped
    else:
        metrics.append(('KMeans-2feat (orig)', 0, 0, 0, 0))
        results_labels['KMeans-2feat (orig)'] = np.zeros(len(X_scaled_df))

    # Method 2 - KMeans k=3 on all features
    k_all = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels_2 = k_all.fit_predict(X_scaled_df)
    centroids_2 = k_all.cluster_centers_
    cluster_scores_2 = [(cid, np.linalg.norm(cen)) for cid, cen in enumerate(centroids_2)]
    sorted_clusters_2 = sorted(cluster_scores_2, key=lambda x: x[1])
    mapping_2 = {cid: rank for rank, (cid, _) in enumerate(sorted_clusters_2)}
    labels_2_mapped = np.array([mapping_2[c] for c in labels_2])
    sil_2 = silhouette_score(X_scaled_df, labels_2_mapped)
    dist_2 = {i: np.mean(labels_2_mapped == i) for i in range(3)}
    metrics.append(('KMeans-all-feat', sil_2, dist_2[0]*100, dist_2[1]*100, dist_2[2]*100))
    results_labels['KMeans-all-feat'] = labels_2_mapped

    # Method 3 - KMeans k=4 on all features
    k4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_3 = k4.fit_predict(X_scaled_df)
    centroids_3 = k4.cluster_centers_
    cluster_scores_3 = [(cid, np.linalg.norm(cen)) for cid, cen in enumerate(centroids_3)]
    sorted_clusters_3 = sorted(cluster_scores_3, key=lambda x: x[1])
    mapping_3 = {
        sorted_clusters_3[0][0]: 0,
        sorted_clusters_3[1][0]: 1,
        sorted_clusters_3[2][0]: 2,
        sorted_clusters_3[3][0]: 2
    }
    labels_3_mapped = np.array([mapping_3[c] for c in labels_3])
    sil_3 = silhouette_score(X_scaled_df, labels_3_mapped)
    dist_3 = {i: np.mean(labels_3_mapped == i) for i in range(3)}
    metrics.append(('KMeans-k4', sil_3, dist_3[0]*100, dist_3[1]*100, dist_3[2]*100))
    results_labels['KMeans-k4'] = labels_3_mapped

    # Method 4 - GMM k=3
    gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    labels_4 = gmm.fit_predict(X_scaled_df)
    centroids_4 = gmm.means_
    cluster_scores_4 = [(cid, np.linalg.norm(cen)) for cid, cen in enumerate(centroids_4)]
    sorted_clusters_4 = sorted(cluster_scores_4, key=lambda x: x[1])
    mapping_4 = {cid: rank for rank, (cid, _) in enumerate(sorted_clusters_4)}
    labels_4_mapped = np.array([mapping_4[c] for c in labels_4])
    sil_4 = silhouette_score(X_scaled_df, labels_4_mapped)
    dist_4 = {i: np.mean(labels_4_mapped == i) for i in range(3)}
    metrics.append(('GMM-k3', sil_4, dist_4[0]*100, dist_4[1]*100, dist_4[2]*100))
    results_labels['GMM-k3'] = labels_4_mapped
    
    print("\n┌──────────────────────┬─────────────┬──────────┬───────────┬───────┐")
    print("│ Method               │ Silhouette  │ Shallow% │ Moderate% │ Deep% │")
    print("├──────────────────────┼─────────────┼──────────┼───────────┼───────┤")
    for m in metrics:
        print(f"│ {m[0]:<20} │   {m[1]:.3f}     │  {m[2]:>4.1f}%   │   {m[3]:>4.1f}%   │ {m[4]:>4.1f}% │")
    print("└──────────────────────┴─────────────┴──────────┴───────────┴───────┘")
    
    # Save comparison plot
    best_method = max(metrics, key=lambda x: x[1])[0]
    plt.figure(figsize=(10,6))
    methods = [m[0] for m in metrics]
    sil_scores = [m[1] for m in metrics]
    sns.barplot(x=methods, y=sil_scores)
    plt.title("Clustering Silhouette Comparison")
    plt.savefig(os.path.join(RESULTS_PATH, 'clustering_comparison.png'))
    plt.close()
    
    return results_labels[best_method]

def generate_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Use KMeans to generate 3 pseudolabels indicating severity."""
    if df.empty:
        return df
        
    print("Generating severity pseudo-labels via KMeans...")
    # Features generally indicative of severity: max depth and area
    # Normalizing features before clustering to prevent bias towards large area values
    features_for_clustering = df[['max_depth', 'pothole_area']].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_for_clustering)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)
    
    # We want to order clusters such that Level 1 (Shallow) < Level 2 (Moderate) < Level 3 (Deep)
    # Calculate a proxy "severity score" for each cluster centroid (e.g., standard distance from origin)
    centroids = kmeans.cluster_centers_
    cluster_scores = [(cluster_id, np.linalg.norm(centroid)) for cluster_id, centroid in enumerate(centroids)]
    
    # Sort clusters by severity score
    sorted_clusters = sorted(cluster_scores, key=lambda x: x[1])
    
    # Create mapping: lowest score -> 0 (Shallow), middle -> 1 (Moderate), highest -> 2 (Deep)
    cluster_mapping = {cluster_id: rank for rank, (cluster_id, score) in enumerate(sorted_clusters)}
    
    # Apply mapping
    df['severity_label'] = [cluster_mapping[c] for c in clusters]
    df['severity_name'] = df['severity_label'].map({0: 'Shallow', 1: 'Moderate', 2: 'Deep'})
    
    # Plot feature distribution colored by assigned severity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pothole_area', y='max_depth', hue='severity_name', palette=['green', 'orange', 'red'])
    plt.title("KMeans Extracted Severity Levels (Max Depth vs Pothole Area)")
    plt.savefig(os.path.join(RESULTS_PATH, 'kmeans_severity_distribution.png'))
    plt.close()
    
    return df

def bootstrap_evaluation(model, X_val, y_val, n_iterations=1000):
    """Evaluate performance repeatedly via bootstrapping."""
    scores = []
    n_size = int(len(X_val))
    
    # Using integer indexing or resetting index to avoid KeyError
    X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
    
    for _ in range(n_iterations):
        # Prepare bootstrap sample
        indices = resample(np.arange(n_size), replace=True, n_samples=n_size)
        X_sample = X_val_np[indices]
        y_sample = y_val_np[indices]
        
        # Evaluate
        predictions = model.predict(X_sample)
        score = accuracy_score(y_sample, predictions)
        scores.append(score)
        
    return scores

def main():
    parser = argparse.ArgumentParser(description="Train ML based pothole severity classifier")
    parser.add_argument("--save_data", action="store_true", default=True, help="Save extracted features to CSV")
    args = parser.parse_args()

    # 1. Feature Extraction
    train_df = extract_dataset_features("train")
    if train_df.empty:
        print("Cannot proceed. Training features extracted are empty. Are labels/depth_maps_1 synced?")
        return
        
    if USE_ADVERSE_AUGMENTATION:
        print("Applying adverse condition augmentation to training set...")
        try:
            import random
            from adverse_conditions import synthesize_condition, get_available_conditions
            # Only use conditions that don't depend on mask context since we're augmenting the whole image
            augment_conditions = ['rain', 'night', 'fog', 'shadow']
            
            # Since generating depth maps on the fly for augmented images requires Depth-Anything,
            # and that would be slow during ML training, we'd ideally pre-compute augmented depth maps.
            # For this pipeline phase, we'll mark the logic structure.
            print(f"  ⚠ Note: Adverse augmentation requires pre-computed depth maps for the variants.")
            print(f"  ⚠ Skipping inline augmentation to avoid blocking ML loop. To use, run augment script first.")
        except ImportError:
            print("  ⚠ adverse_conditions.py not found. Skipping augmentation.")

    val_df = extract_dataset_features("valid")
    
    # 2. Pseudo Label Generation (Fit on train, apply to valid mapping logically)
    # We will fit KMeans on train data to define the cluster mapping standard.
    train_df = generate_pseudo_labels(train_df)
    
    # Apply identical transformation to val_df using a simple Nearest-Centroid or 
    # train another instance but consistency matters. So re-using same approach globally or:
    # the simplest robust way: apply the same clustering process to the full data then split it back,
    # OR map valid data to the closest center found during training.
    # For simplicity, cluster valid set independently if it's large enough and follows same distribution.
    # To maintain strict separation, we determine closest centroid from training data.
    if not val_df.empty:
        print("Generating pseudo labels for validation set mapping...")
        features_train_clu = train_df[['max_depth', 'pothole_area']].copy()
        features_val_clu = val_df[['max_depth', 'pothole_area']].copy()
        
        sc = StandardScaler()
        sc.fit(features_train_clu)
        
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        km.fit(sc.transform(features_train_clu))
        
        centroids = km.cluster_centers_
        cluster_scores = [(cluster_id, np.linalg.norm(cen)) for cluster_id, cen in enumerate(centroids)]
        sorted_clusters = sorted(cluster_scores, key=lambda x: x[1])
        cluster_mapping = {cluster_id: rank for rank, (cluster_id, score) in enumerate(sorted_clusters)}
        
        # Predict on valid
        val_clusters = km.predict(sc.transform(features_val_clu))
        val_df['severity_label'] = [cluster_mapping[c] for c in val_clusters]
        val_df['severity_name'] = val_df['severity_label'].map({0: 'Shallow', 1: 'Moderate', 2: 'Deep'})

    if args.save_data:
        train_df.to_csv(os.path.join(RESULTS_PATH, "train_features.csv"), index=False)
        if not val_df.empty:
            val_df.to_csv(os.path.join(RESULTS_PATH, "valid_features.csv"), index=False)
            
    # 3. Model Training
    print("\nPreparing models...")
    
    if USE_EXTENDED_FEATURES:
        feature_cols = [
            'height', 'width', 'box_area', 'pothole_area', 'nonpothole_area',
            'mean_depth', 'max_depth', 'min_depth', 'depth_std', 'depth_range', 'p90_depth',
            'aspect_ratio', 'solidity', 'compactness', 'depth_skewness',
            'depth_kurtosis', 'boundary_gradient', 'weighted_mean_depth',
            'surface_area_px2', 'surface_area_cm2'
        ]
        # ── Phase 1 geometry features ────────────────────────────────────
        # These are extracted by extract_all_geometry_features() during
        # feature building, but until now were never selected for training —
        # so every model shipped was trained on depth statistics alone and had
        # never seen the depth-independent geometry the project is built on.
        if USE_GEOMETRY_FEATURES:
            feature_cols += GEOMETRY_FEATURE_COLS
    else:
        feature_cols = [
            'height', 'width', 'box_area', 'pothole_area', 'nonpothole_area', 
            'mean_depth', 'max_depth', 'min_depth', 'depth_std', 'depth_range', 'p90_depth'
        ]
    
    # Keep only available columns
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    target_col = 'severity_label'
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    if USE_REAL_LABELS:
        y_train_arr, label_sources = load_real_labels(train_df, SEVERITY_LABELS_CSV)
        train_df['severity_label'] = y_train_arr
        y_train = train_df[target_col]
        if not val_df.empty:
            X_val = val_df[feature_cols]
            y_val_arr, _ = load_real_labels(val_df, SEVERITY_LABELS_CSV)
            val_df['severity_label'] = y_val_arr
            y_val = val_df[target_col]
        else:
            from sklearn.model_selection import train_test_split
            print("No validation set found, splitting train set...")
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    else:
        if not val_df.empty:
            X_val = val_df[feature_cols]
            y_val = val_df[target_col]
        else:
            # Fallback if no validation data
            from sklearn.model_selection import train_test_split
            print("No validation set found, splitting train set...")
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Standardize features (crucial for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    joblib.dump(scaler, os.path.join(MODELS_PATH, "feature_scaler.pkl"))

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
        "SVM": SVC(random_state=42, probability=True, class_weight='balanced'),
        "Naive Bayes": GaussianNB(),
        "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss', scale_pos_weight=3),
        "lightgbm": LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "ensemble": VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
                ('xgb', XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')),
                ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1)),
            ],
            voting='soft'
        )
    }

    results = []
    trained_models = {}

    for name, mm in models.items():
        print(f"\nTraining {name}...")
        mm.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(mm, os.path.join(MODELS_PATH, f"{name.replace(' ', '_').lower()}.pkl"))
        trained_models[name] = mm
        
        # Predict & Evaluate metrics
        y_pred = mm.predict(X_val_scaled)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro F1": report.get('macro avg', {}).get('f1-score', 0)
        })
        
        print(f"[{name}] Accuracy: {acc:.4f}")

    # Save scaler + all models immediately after training
    joblib.dump(scaler, os.path.join(MODELS_PATH, "feature_scaler.pkl"))
    print(f"\nAll models and scaler saved to {MODELS_PATH}")
        
    # 4. Evaluation and Bootstrapping for Best Model Selection
    print("\nStarting bootstrapping evaluation (n=100 iterations)...")
    bootstrap_results = {}
    
    for name, mm in trained_models.items():
        # Retrieve scores robustly
        b_scores = bootstrap_evaluation(mm, X_val_scaled, y_val, 100)
        bootstrap_results[name] = b_scores
        
        mean_acc = np.mean(b_scores)
        ci_lower = np.percentile(b_scores, 2.5)
        ci_upper = np.percentile(b_scores, 97.5)
        
        print(f"{name} Bootstrap 95% CI Accuracy: {mean_acc:.4f} ({ci_lower:.4f} - {ci_upper:.4f})")
        
        # Update results
        for r in results:
            if r["Model"] == name:
                r["Bootstrap Mean Acc"] = mean_acc

    # Plot performance comparison
    res_df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(res_df.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=res_df, x="Model", y="Accuracy")
    plt.title("Classifier Accuracy Comparison")
    plt.ylim(0, 1)
    plt.savefig(os.path.join(RESULTS_PATH, "accuracy_comparison.png"))
    plt.close()
    
    # Plot Bootstrapping distributions
    plt.figure(figsize=(10, 6))
    for name, scores in bootstrap_results.items():
        sns.kdeplot(scores, label=name)
    plt.title("Bootstrapped Accuracy Distribution")
    plt.xlabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(RESULTS_PATH, "bootstrap_distributions.png"))
    plt.close()
    
    # Run enhanced evaluation
    enhanced_evaluation(trained_models, X_train_scaled, X_val_scaled, y_train, y_val, feature_cols)
    print("All extended models saved to " + MODELS_DIR)

def enhanced_evaluation(models_dict, X_train_scaled, X_val_scaled, y_train, y_val, feature_names):
    # EVALUATION 1 - Confusion Matrix
    print("\nRunning enhanced evaluations...")
    for name, model in models_dict.items():
        cm = confusion_matrix(y_val, model.predict(X_val_scaled))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Shallow', 'Moderate', 'Deep'])
        plt.figure(figsize=(8,6))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f"Confusion Matrix — {name}")
        plt.savefig(os.path.join(RESULTS_PATH, f"confusion_matrix_{name}.png"))
        plt.close('all')

    # EVALUATION 2 - Per-class F1
    for name, model in models_dict.items():
        report = classification_report(y_val, model.predict(X_val_scaled), target_names=['Shallow', 'Moderate', 'Deep'])
        print(f"\nClassification Report for {name}:\n{report}")
        with open(os.path.join(RESULTS_PATH, f"classification_report_{name}.txt"), 'w') as f:
            f.write(report)

    # EVALUATION 3 - Calibration curves
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val_scaled)
            for class_idx in range(3):
                y_binary = (y_val == class_idx).astype(int)
                fraction_pos, mean_pred = calibration_curve(y_binary, proba[:, class_idx], n_bins=10)
                ax1.plot(mean_pred, fraction_pos, "s-", label=f"{name} (class {class_idx})")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration — All Models")
    ax1.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "calibration_curves.png"))
    plt.close('all')

    # EVALUATION 4 - SHAP feature importance
    tree_models = ['Random Forest', 'xgboost', 'lightgbm', 'ensemble']
    for name, model in models_dict.items():
        try:
            if name in tree_models:
                # Need to use the underlying model for voting classifier or tree explainer
                if name == 'ensemble':
                    continue # SHAP complex for voting classifier, skip
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_val_scaled)
            else:
                background = shap.sample(X_train_scaled, 50)
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_val_scaled[:50])
                
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list): # Multi-class
                shap.summary_plot(shap_values, X_val_scaled[:50] if not name in tree_models else X_val_scaled, feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_val_scaled[:50] if not name in tree_models else X_val_scaled, feature_names=feature_names, show=False)
            plt.savefig(os.path.join(RESULTS_PATH, f"shap_{name}.png"), bbox_inches='tight')
            plt.close('all')
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_val_scaled[:50] if not name in tree_models else X_val_scaled, feature_names=feature_names, plot_type="bar", show=False)
            plt.savefig(os.path.join(RESULTS_PATH, f"shap_bar_{name}.png"), bbox_inches='tight')
            plt.close('all')
        except Exception as e:
            print(f"Skipping SHAP for {name} due to error: {e}")

    # EVALUATION 5 - Learning curves
    for name in ['Random Forest', 'xgboost']:
        if name in models_dict:
            model_class = models_dict[name].__class__
            model_params = models_dict[name].get_params()
            train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
            train_scores = []
            val_scores = []
            for fraction in train_sizes:
                n_samples = int(len(X_train_scaled) * fraction)
                indices = np.random.RandomState(42).choice(len(X_train_scaled), n_samples, replace=False)
                X_sub = X_train_scaled[indices]
                y_sub = np.array(y_train)[indices]
                
                fresh_model = model_class(**model_params)
                fresh_model.fit(X_sub, y_sub)
                
                train_scores.append(accuracy_score(y_sub, fresh_model.predict(X_sub)))
                val_scores.append(accuracy_score(y_val, fresh_model.predict(X_val_scaled)))
                
            plt.figure()
            plt.plot(train_sizes, train_scores, 'o-', label="Train")
            plt.plot(train_sizes, val_scores, 'o-', label="Validation")
            plt.title(f"Learning Curve — {name}")
            plt.xlabel("Training Set Fraction")
            plt.ylabel("Accuracy")
            plt.legend(loc="best")
            plt.savefig(os.path.join(RESULTS_PATH, f"learning_curve_{name}.png"))
            plt.close('all')

    # EVALUATION 6 - Ablation study
    if 'Random Forest' in models_dict:
        geometric_features = [
            'height', 'width', 'box_area', 'pothole_area', 'nonpothole_area',
            'aspect_ratio', 'solidity', 'compactness',
            'surface_area_px2', 'surface_area_cm2'
        ]
        depth_features = [
            'mean_depth', 'max_depth', 'min_depth', 'depth_std',
            'depth_range', 'p90_depth', 'depth_skewness', 'depth_kurtosis',
            'boundary_gradient', 'weighted_mean_depth'
        ]

        # --- Geometry-based curvature features (novel extension) ---
        # These are computed from mask shape only, no depth required.
        curvature_feature_names = [
            'max_curvature', 'mean_curvature', 'std_curvature',
            'p90_curvature', 'high_curvature_fraction', 'curvature_entropy',
            'concave_fraction', 'curvature_sign_changes', 'contour_length',
            'contour_elongation',
            'mean_bowl_depth', 'max_bowl_depth', 'std_bowl_depth',
            'mean_road_curvature', 'slope_variance',
            'mean_normal_deviation', 'max_normal_deviation',
            'std_normal_deviation', 'p90_normal_deviation',
        ]

        # Filter to features that actually exist in the data
        available_curvature = [c for c in curvature_feature_names if c in feature_names]

        feature_sets = {
            'Geometric Only': [c for c in geometric_features if c in feature_names],
            'Depth Only': [c for c in depth_features if c in feature_names],
            'Curvature Only': available_curvature,
            'All Features': [c for c in feature_names if c not in ('img_name', 'pothole_idx', 'severity_label')],
        }
        
        ablation_results = []
        df_train = pd.DataFrame(X_train_scaled, columns=feature_names)
        df_val = pd.DataFrame(X_val_scaled, columns=feature_names)
        
        for subset_name, columns in feature_sets.items():
            if not columns:
                print(f"  Skipping ablation for '{subset_name}': no matching features found")
                continue
            X_tr_sub = df_train[columns]
            X_vl_sub = df_val[columns]
            
            rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            rf.fit(X_tr_sub, y_train)
            
            y_pred = rf.predict(X_vl_sub)
            acc = accuracy_score(y_val, y_pred)
            from sklearn.metrics import f1_score
            mac_f1 = f1_score(y_val, y_pred, average='macro')
            
            ablation_results.append({
                'Feature Subset': subset_name,
                'Num Features': len(columns),
                'Val Accuracy': acc,
                'Macro F1': mac_f1
            })
            
        ablation_df = pd.DataFrame(ablation_results)
        print(f"\nAblation Study (4-config):\n{ablation_df}")
        ablation_df.to_csv(os.path.join(RESULTS_PATH, 'ablation_study.csv'), index=False)

        # Also save a separate geometry ablation file for the API
        geometry_ablation = ablation_df.copy()
        geometry_ablation.to_csv(os.path.join(RESULTS_PATH, 'geometry_ablation.csv'), index=False)

        # --- Train geometry-only models for /analyze/geometry endpoint ---
        if available_curvature:
            print("\n=== Training Geometry-Only Models ===")
            # MODELS_PATH already ends in ml_models[/extended]; joining "ml_models"
            # again produced ml_models/ml_models/geometry_only.
            geometry_models_dir = os.path.join(SCRIPT_DIR, "ml_models", "geometry_only")
            os.makedirs(geometry_models_dir, exist_ok=True)

            X_train_geo = df_train[available_curvature].values
            X_val_geo = df_val[available_curvature].values

            geo_scaler = StandardScaler()
            X_train_geo_scaled = geo_scaler.fit_transform(X_train_geo)
            X_val_geo_scaled = geo_scaler.transform(X_val_geo)

            # Save geometry-only scaler
            joblib.dump(geo_scaler, os.path.join(geometry_models_dir, "feature_scaler.pkl"))

            # Train a subset of the models on geometry-only features
            geo_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200, class_weight='balanced', random_state=42, max_depth=10
                ),
                'svm': SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True),
                'logistic_regression': LogisticRegression(
                    max_iter=1000, class_weight='balanced', random_state=42
                ),
            }

            for name, model in geo_models.items():
                model.fit(X_train_geo_scaled, y_train)
                geo_acc = accuracy_score(y_val, model.predict(X_val_geo_scaled))
                print(f"  Geometry-only {name}: {geo_acc:.4f}")
                joblib.dump(model, os.path.join(geometry_models_dir, f"{name}.pkl"))

            # Save feature names list for inference
            joblib.dump(available_curvature, os.path.join(geometry_models_dir, "feature_names.pkl"))
            print(f"  Geometry-only models saved to {geometry_models_dir}")

    # EVALUATION 7 - Feature correlation heatmap
    df_all_feats = pd.DataFrame(X_train_scaled, columns=feature_names)
    corr = df_all_feats.corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(RESULTS_PATH, "feature_correlation.png"))
    plt.close('all')
    
    # Print high correlations
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.9:
                print(f"WARNING: High correlation between {corr.columns[i]} and {corr.columns[j]}: {corr.iloc[i, j]:.2f}")

    # EVALUATION 8 - t-SNE / UMAP
    from sklearn.manifold import TSNE
    colors = {0: 'yellow', 1: 'orange', 2: 'red'}
    num_val_samples = len(X_val_scaled)
    if num_val_samples > 1:
        tsne_perplexity = max(1, min(30, num_val_samples - 1))
        tsne_common_kwargs = {
            "n_components": 2,
            "perplexity": tsne_perplexity,
            "random_state": 42
        }
        try:
            # Newer scikit-learn versions use max_iter.
            tsne = TSNE(max_iter=1000, **tsne_common_kwargs)
        except TypeError:
            # Older scikit-learn versions use n_iter.
            tsne = TSNE(n_iter=1000, **tsne_common_kwargs)

        X_2d = tsne.fit_transform(X_val_scaled)
        plt.figure(figsize=(8, 6))
        for class_idx in range(3):
            masker = (np.array(y_val) == class_idx)
            plt.scatter(X_2d[masker, 0], X_2d[masker, 1], c=colors[class_idx], label=['Shallow', 'Moderate', 'Deep'][class_idx], alpha=0.7)
        plt.title("t-SNE of Feature Space (colored by severity)")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, "tsne_features.png"))
        plt.close('all')
    else:
        print("Skipping t-SNE: need at least 2 validation samples.")
    
    try:
        import umap
        reducer = umap.UMAP(random_state=42)
        X_umap = reducer.fit_transform(X_val_scaled)
        plt.figure(figsize=(8, 6))
        for class_idx in range(3):
            masker = (np.array(y_val) == class_idx)
            plt.scatter(X_umap[masker, 0], X_umap[masker, 1], c=colors[class_idx], label=['Shallow', 'Moderate', 'Deep'][class_idx], alpha=0.7)
        plt.title("UMAP of Feature Space (colored by severity)")
        plt.legend()
        plt.savefig(os.path.join(RESULTS_PATH, "umap_features.png"))
        plt.close('all')
    except ImportError:
        print("UMAP not installed — skipping. pip install umap-learn")

    # EVALUATION 9 - Updated accuracy comparison plot
    accuracies = []
    names = []
    for name, model in models_dict.items():
        acc = accuracy_score(y_val, model.predict(X_val_scaled))
        accuracies.append(acc)
        names.append(name)
        
    df_acc = pd.DataFrame({'Model': names, 'Accuracy': accuracies}).sort_values('Accuracy', ascending=False)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_acc, x='Accuracy', y='Model', palette='YlOrBr_r')
    plt.title("Accuracy Comparison (All Models)")
    plt.xlim(0, 1)
    plt.savefig(os.path.join(RESULTS_PATH, "accuracy_comparison_all_models.png"))
    plt.style.use('default')
    plt.close('all')

if __name__ == "__main__":
    main()
