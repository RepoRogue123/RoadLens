"""
Segmentation training set v2: real outlines only, splits that cannot leak.

What changes from the production model's data
---------------------------------------------
Production YOLO (`yolo-segmentation/model/best.pt`) was fine-tuned on the Kaggle
set alone — about 300 photographs — and finds 70.8% of the RealSense-measured
PothRGBD potholes. Every pothole it misses gets no depth estimate at all.

This set adds the two other sources that carry true pothole OUTLINES:

    kaggle     data1/            640x640, Roboflow polygons
    p600       pothole600/       400x400 stereo frames, binary masks -> polygons
    pothrgbd   archive/PUBLIC POTHOLE DATASET/   640x480 RealSense frames, polygons

and deliberately leaves out RDD2022 and the GPS set: their labels are boxes that
were written as rectangle polygons, and 76% of merged_dataset is such
rectangles. Training a segmenter on them teaches it that potholes are boxes.

Splits
------
    kaggle     data1/train -> train/val (10% by source_stem);  data1/valid -> test_kaggle
               (data1/valid was also the production model's validation set, never its
               training set, so both models can be scored on it)
    p600       training -> train;  validation -> val;  testing -> test_p600.
               Images in val/test with a perceptual-hash near-duplicate in ANY training
               image are dropped (9 found).
    pothrgbd   split by CAPTURE SESSION, not frame: a gap of more than 60 s starts a
               new session (228 sessions). 29% of frames follow the previous one within
               10 s, so frame-level splits could put two shots of one pothole on both
               sides. 70% of sessions train, 10% val, 20% test_pothrgbd.

The PothRGBD test sessions are written to test_pothrgbd_keys.txt; the depth model's
downstream check must only use those frames, because the new segmenter has seen the rest.

Output: archive/yolo_seg_v2/ (gitignored) with one data yaml per evaluation set.

Usage:
    python scripts/build_yolo_seg_v2.py
"""
import glob
import os
import shutil
import sys
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from dataset_registry import source_stem                                  # noqa: E402
from scripts.build_provenance_index import hamming_matrix, phash          # noqa: E402
from scripts.pothrgbd_metric_labels import (                              # noqa: E402
    DATA_DIR, capture_sessions, timestamp_key,
)

OUT = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
SEED = 20260925
MIN_POLY_PX = 100          # same floor as segmentation.get_all_masks(min_area=100)
NEAR_DUP = 6


def mask_to_polygons(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    h, w = m.shape
    cnts, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in cnts:
        if cv2.contourArea(c) < MIN_POLY_PX:
            continue
        pts = c.reshape(-1, 2).astype(np.float64) / [w, h]
        if len(pts) >= 3:
            lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in np.clip(pts, 0, 1)))
    return lines


def put(split, name, image_path, label_lines):
    d_img = os.path.join(OUT, split, "images")
    d_lab = os.path.join(OUT, split, "labels")
    os.makedirs(d_img, exist_ok=True)
    os.makedirs(d_lab, exist_ok=True)
    ext = os.path.splitext(image_path)[1]
    shutil.copy2(image_path, os.path.join(d_img, name + ext))
    with open(os.path.join(d_lab, name + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + ("\n" if label_lines else ""))


def read_label(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if len(ln.split()) >= 7]     # class + >= 3 points


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    rng = np.random.default_rng(SEED)
    counts = {}

    def add(split, *args):
        put(split, *args)
        counts[split] = counts.get(split, 0) + 1

    train_hashes = []

    # ── Kaggle ──
    tr = sorted(glob.glob(os.path.join(PROJECT_DIR, "data1", "train", "images", "*")))
    stems = sorted({source_stem(os.path.basename(p)) for p in tr})
    val_stems = set(rng.permutation(stems)[: max(1, len(stems) // 10)])
    for p in tr:
        lab = read_label(os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                                      os.path.splitext(os.path.basename(p))[0] + ".txt"))
        if lab is None:
            continue
        split = "val" if source_stem(os.path.basename(p)) in val_stems else "train"
        add(split, "kaggle_" + os.path.splitext(os.path.basename(p))[0], p, lab)
        if split == "train":
            train_hashes.append(phash(p))
    for p in sorted(glob.glob(os.path.join(PROJECT_DIR, "data1", "valid", "images", "*"))):
        lab = read_label(os.path.join(os.path.dirname(os.path.dirname(p)), "labels",
                                      os.path.splitext(os.path.basename(p))[0] + ".txt"))
        if lab is not None:
            add("test_kaggle", "kaggle_" + os.path.splitext(os.path.basename(p))[0], p, lab)

    # ── PothRGBD, split by capture session ──
    imgs = sorted(glob.glob(os.path.join(DATA_DIR, "images", "*.jpg")))
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    keys = [timestamp_key(p) for p in imgs]
    sess_of = capture_sessions(keys)
    session = np.array([sess_of[k] for k in keys])
    # Whole sessions are assigned until test holds 20% of FRAMES and val 10%.
    # (Sessions run from 1 to 71 frames, so a 20% share of sessions can be a 7% share of frames.)
    ids = rng.permutation(np.unique(session))
    n = len(ids)
    sizes = np.bincount(session)
    test_s, val_s, filled = set(), set(), 0
    for s in ids:
        if filled < 0.2 * len(keys):
            test_s.add(s)
        elif filled < 0.3 * len(keys):
            val_s.add(s)
        else:
            break
        filled += sizes[s]
    test_keys = []
    for p, k, s in zip(imgs, keys, session):
        lab = read_label(labs.get(k, ""))
        if lab is None:
            continue
        split = "test_pothrgbd" if s in test_s else "val" if s in val_s else "train"
        add(split, f"pothrgbd_{k}", p, lab)
        if split == "test_pothrgbd":
            test_keys.append(k)
        elif split == "train":
            train_hashes.append(phash(p))

    # ── Pothole-600 (training last, so its hashes join before val/test are screened) ──
    p6 = {s: sorted(glob.glob(os.path.join(PROJECT_DIR, "pothole600", s, "rgb", "*.png")))
          for s in ("training", "validation", "testing")}
    for p in p6["training"]:
        lab = mask_to_polygons(p.replace(os.sep + "rgb" + os.sep, os.sep + "label" + os.sep)
                               .replace("/rgb/", "/label/"))
        if lab is None:
            continue
        add("train", f"p600_train_{os.path.splitext(os.path.basename(p))[0]}", p, lab)
        train_hashes.append(phash(p))
    th = np.array(train_hashes, dtype=np.uint64)
    dropped = 0
    for src, split in (("validation", "val"), ("testing", "test_p600")):
        for p in p6[src]:
            h = np.array([phash(p)], dtype=np.uint64)
            if hamming_matrix(h, th).min() <= NEAR_DUP:
                dropped += 1
                continue
            lab = mask_to_polygons(p.replace(os.sep + "rgb" + os.sep, os.sep + "label" + os.sep)
                                   .replace("/rgb/", "/label/"))
            if lab is None:
                continue
            add(split, f"p600_{src}_{os.path.splitext(os.path.basename(p))[0]}", p, lab)

    with open(os.path.join(OUT, "test_pothrgbd_keys.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(test_keys)) + "\n")

    root = OUT.replace("\\", "/")
    for name, val in (("data", "val"), ("test_kaggle", "test_kaggle"),
                      ("test_p600", "test_p600"), ("test_pothrgbd", "test_pothrgbd")):
        with open(os.path.join(OUT, f"{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {root}\ntrain: train/images\nval: {val}/images\nnames:\n  0: pothole\n")

    print("images per split:", counts)
    print(f"PothRGBD: {n} capture sessions -> {len(test_s)} test, {len(val_s)} val")
    print(f"Pothole-600 val/test images dropped as near-duplicates of training images: {dropped}")
    print(f"wrote {os.path.relpath(OUT, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
