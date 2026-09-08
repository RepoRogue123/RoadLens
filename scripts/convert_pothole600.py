import os
import shutil
import csv
import json
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_DIR = PROJECT_ROOT / "merged_dataset"
STATS_DIR = MERGED_DIR / "dataset_stats"
REPORT_CSV = STATS_DIR / "dataset_report.csv"
SEVERITY_DIR = MERGED_DIR / "severity_labels"
STEREO_DIR = MERGED_DIR / "stereo"

# ── Stereo calibration status ────────────────────────────────────────────
# Pothole-600 ships RGB + transformed-disparity + segmentation labels, but no
# camera calibration file. Without the true focal length and stereo baseline
# the disparity->depth conversion cannot produce metric units.
#
# Set CALIBRATION_AVAILABLE = True and fill in the real values to obtain metric
# depth; until then the output is relative (inverse disparity, arbitrary scale)
# and the output directory is named accordingly so nothing downstream mistakes
# it for a measurement in metres.
CALIBRATION_AVAILABLE = False
FOCAL_LENGTH_PX = 1.0    # placeholder — real value required for metric output
BASELINE_M = 1.0         # placeholder — real value required for metric output

RELATIVE_DEPTH_DIRNAME = "metric_depth" if CALIBRATION_AVAILABLE else "relative_depth"

# Sub-directories setup
SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
for split in ["train", "valid", "test"]:
    (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (STEREO_DIR / "disparity" / split).mkdir(parents=True, exist_ok=True)
    (STEREO_DIR / RELATIVE_DEPTH_DIRNAME / split).mkdir(parents=True, exist_ok=True)


def append_to_csv(rows):
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def find_pothole600_dir():
    # Since the user handles the manual download, we expect the folder in root
    possible_names = ["pothole-600", "Pothole600", "pothole600", "Pothole-600"]
    for name in possible_names:
        path = PROJECT_ROOT / name
        if path.exists() and path.is_dir():
            return path
    # Generic search for anything mentioning pothole and 600
    for path in PROJECT_ROOT.iterdir():
        if path.is_dir() and "pothole" in path.name.lower() and "600" in path.name:
            return path
    return None

def process_masks_to_yolo(mask_path, w, h):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []
    
    # Binarize in case it's smooth
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue
        cnt = cnt.squeeze()
        if len(cnt.shape) == 1:
            cnt = cnt.reshape(-1, 2)
            
        coords = []
        for point in cnt:
            x, y = point[0] / w, point[1] / h
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            coords.append(f"{x:.6f} {y:.6f}")
            
        if len(coords) >= 3: # polygon requires at least 3 points
            polygons.append("0 " + " ".join(coords))
    return polygons

def process_disparity(disp_path, dest_disp_path, dest_depth_path, focal_length, baseline):
    """
    Convert a Pothole-600 disparity map into a depth map.

    IMPORTANT — the output is only metric when `focal_length` and `baseline`
    are the real calibration values of the capture rig. Pothole-600 ships no
    calibration file (see CALIBRATION_AVAILABLE below), so with the placeholder
    values of 1.0 this returns *inverse disparity in arbitrary units*, which is
    proportional to depth but carries no scale. It is usable as a relative
    depth reference; it must not be quoted in centimetres or metres.
    """
    disp = cv2.imread(str(disp_path), cv2.IMREAD_UNCHANGED)
    if disp is None:
        return False

    disp_float = disp.astype(np.float32)
    # Z = f * B / d   — metric only if f and B are the true rig parameters.
    depth = (focal_length * baseline) / (disp_float + 1e-8)

    np.save(str(dest_disp_path), disp_float)
    np.save(str(dest_depth_path), depth)
    return True

def convert_pothole600():
    p600_dir = find_pothole600_dir()
    if p600_dir is None:
        print("Pothole-600 dataset folder not found in the root directory.")
        print("Please manually download and extract it to a folder like 'Pothole600' in the project root.")
        return

    print(f"Found Pothole-600 dataset at: {p600_dir.name}")
    
    # 1. Look for rgb images
    left_images = []
    rgb_dirs = list(p600_dir.rglob("rgb"))
    if rgb_dirs:
        for rgb_dir in rgb_dirs:
            left_images.extend(rgb_dir.glob("*.jpg"))
            left_images.extend(rgb_dir.glob("*.png"))
    else:
        left_images = list(p600_dir.rglob("*left*.jpg")) + list(p600_dir.rglob("*left*.png"))
        if not left_images:
             # Fallback to general images if 'left' isn't explicitly in the name but usually it is organized in folders
             left_dir = list(p600_dir.rglob("*left*"))
             if left_dir and left_dir[0].is_dir():
                 left_images = list(left_dir[0].glob("*.jpg")) + list(left_dir[0].glob("*.png"))
             else:
                 print("Could not specifically identify 'left' or 'rgb' camera images. Processing all images...")
                 left_images = list(p600_dir.rglob("*.jpg")) + list(p600_dir.rglob("*.png"))
    
    print(f"Found {len(left_images)} rgb/left camera images.")

    # 2. Extract severity labels
    severity_map = {} # img_stem -> label_int
    label_map = {"shallow": 0, "moderate": 1, "deep": 2}
    
    annotations_files = list(p600_dir.rglob("*.csv")) + list(p600_dir.rglob("*.txt")) + list(p600_dir.rglob("*.json"))
    for ann_file in annotations_files:
        try:
            content = ann_file.read_text(encoding='utf-8').lower()
            if "shallow" in content or "moderate" in content or "deep" in content:
                # Naive parse lines
                lines = content.split('\n')
                for line in lines:
                    line = line.replace('"', '').replace(',', ' ').strip()
                    parts = line.split()
                    if not parts: continue
                    img_id = parts[0].replace('.jpg', '').replace('.png', '')
                    for k, v in label_map.items():
                        if k in line:
                            severity_map[img_id] = v
                            break
        except Exception as e:
            pass
            
    print(f"Extracted {len(severity_map)} severity labels.")

    # 3. Process images
    labeled_valid = []
    unlabeled_valid = []
    focal_length = FOCAL_LENGTH_PX
    baseline = BASELINE_M
    if not CALIBRATION_AVAILABLE:
        print("  " + "!" * 68)
        print("  !  NO STEREO CALIBRATION AVAILABLE for Pothole-600.")
        print("  !  Disparity is being converted with placeholder f=B=1.0, so the")
        print("  !  output is INVERSE DISPARITY IN ARBITRARY UNITS, not metres.")
        print(f"  !  Writing to stereo/{RELATIVE_DEPTH_DIRNAME}/ (not 'metric_depth').")
        print("  !  Do not quote these values in cm/m. To make them metric, set")
        print("  !  CALIBRATION_AVAILABLE=True with the real focal length/baseline.")
        print("  " + "!" * 68)
    
    # try to parse readme for focal_length and baseline
    readme_files = list(p600_dir.rglob("*readme*")) + list(p600_dir.rglob("*.txt"))
    for rf in readme_files:
        try:
           content = rf.read_text().lower()
           # Naive extraction - in a real scenario we'd use regex based on expected format
           # Using defaults for now to ensure pipeline runs
        except:
           pass

    for img_path in tqdm(left_images, desc="Scanning Pothole-600 images"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        stem = img_path.stem
        # check for masks
        mask_path = img_path.parent / f"{stem}_mask.png"
        if not mask_path.exists():
            # try sibling mask dir
            sibling_mask = img_path.parent.parent / "masks" / f"{stem}.png"
            sibling_label = img_path.parent.parent / "label" / f"{stem}.png"
            if sibling_mask.exists():
                mask_path = sibling_mask
            elif sibling_label.exists():
                mask_path = sibling_label
                
        polygons = []
        if mask_path.exists():
            polygons = process_masks_to_yolo(mask_path, w, h)
            
        # check for disparity
        disp_path = img_path.parent / f"{stem}_disp.png"
        if not disp_path.exists():
             sibling_disp = img_path.parent.parent / "disparity" / f"{stem}.png"
             sibling_tdisp = img_path.parent.parent / "tdisp" / f"{stem}.png"
             if sibling_disp.exists(): disp_path = sibling_disp
             elif sibling_tdisp.exists(): disp_path = sibling_tdisp
             elif (img_path.parent.parent / "disparity" / f"{stem}.pfm").exists():
                 disp_path = img_path.parent.parent / "disparity" / f"{stem}.pfm"
                 
        has_disp = disp_path.exists()
        
        info = {
            "img": img_path,
            "polygons": polygons,
            "has_disp": has_disp,
            "disp_path": disp_path if has_disp else None,
            "w": w,
            "h": h,
            "severity": severity_map.get(stem, None)
        }
        
        if info["severity"] is not None:
            labeled_valid.append(info)
        else:
            unlabeled_valid.append(info)
            

    # Split and save
    random.shuffle(labeled_valid)
    random.shuffle(unlabeled_valid)
    
    def process_split(data_list, train_pct, val_pct):
        n = len(data_list)
        train_idx = int(train_pct * n)
        val_idx = int((train_pct + val_pct) * n)
        
        csv_rows = []
        severity_rows = []
        
        for i, info in enumerate(data_list):
            if i < train_idx: split = "train"
            elif i < val_idx: split = "valid"
            else: split = "test"
            
            orig_name = info["img"].name
            new_name = f"p600_left_{orig_name}"
            new_stem = Path(new_name).stem
            
            shutil.copy2(info["img"], MERGED_DIR / split / "images" / new_name)
            if info["polygons"]:
                lbl_dest = MERGED_DIR / split / "labels" / f"{new_stem}.txt"
                with open(lbl_dest, "w") as f:
                    f.write("\n".join(info["polygons"]))
                    
            if info["has_disp"]:
                disp_dest = STEREO_DIR / "disparity" / split / f"{new_stem}.npy"
                depth_dest = STEREO_DIR / RELATIVE_DEPTH_DIRNAME / split / f"{new_stem}.npy"
                process_disparity(info["disp_path"], disp_dest, depth_dest, focal_length, baseline)
                
            has_label = len(info["polygons"]) > 0
            csv_rows.append([new_name, "pothole600", split, has_label, info["w"], info["h"], "ok", "", ""])
            
            if info["severity"] is not None:
                rev_map = {0: "Shallow", 1: "Moderate", 2: "Deep"}
                sev_label = rev_map.get(info["severity"], "Unknown")
                severity_rows.append([new_name, info["severity"], sev_label, info["has_disp"]])
                
        return csv_rows, severity_rows

    print("Writing labeled split...")
    c1, s1 = process_split(labeled_valid, 0.70, 0.15)
    print("Writing unlabeled split...")
    c2, s2 = process_split(unlabeled_valid, 0.80, 0.10)
    
    append_to_csv(c1 + c2)
    
    sev_csv_path = SEVERITY_DIR / "pothole600_annotations.csv"
    file_exists = sev_csv_path.exists()
    with open(sev_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["image_name", "severity_int", "severity_label", "has_disparity"])
        writer.writerows(s1 + s2)

if __name__ == "__main__":
    convert_pothole600()
    print("Pothole-600 processing complete.")
