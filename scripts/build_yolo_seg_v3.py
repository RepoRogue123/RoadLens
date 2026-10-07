"""
Segmentation training set v3 = v2 + what RDD2022 and Mendeley can add.

v2 (archive/yolo_seg_v2) holds every true outline in the project. v3 adds, to TRAINING
only, the frames chosen by scripts/build_corpus_manifest.py:

    negatives     RDD dashcam frames with no pothole box, written with an empty label
    box outlines  RDD and Mendeley pothole frames whose every box SAM 2 turned into an
                  accepted outline (scripts/boxes_to_polygons.py)

Validation and the three test sets stay exactly v2's: true outlines only, so the two
models are scored on the same thing and no SAM-made outline ever grades a model.

Two training lists, so the two additions can be judged separately:

    data_neg.yaml    v2 + negatives
    data_full.yaml   v2 + negatives + box outlines

Frames wider than 1280 px are stored at 1280 (Norway's are 4040).

Usage:
    python scripts/build_yolo_seg_v3.py
"""
import glob
import os
import shutil
import sys

import cv2
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(PROJECT_DIR, "archive")
V2 = os.path.join(ARCHIVE, "yolo_seg_v2")
CORPUS = os.path.join(ARCHIVE, "corpus_v3")
OUT = os.path.join(ARCHIVE, "yolo_seg_v3")
MAX_SIDE = 1280


def put(name, src_image, label_text):
    dst = os.path.join(OUT, "extra", "images", name + ".jpg")
    img = cv2.imread(src_image)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        shutil.copy2(src_image, dst)
    with open(os.path.join(OUT, "extra", "labels", name + ".txt"), "w", encoding="utf-8") as f:
        f.write(label_text)
    return dst.replace("\\", "/")


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for d in ("images", "labels"):
        os.makedirs(os.path.join(OUT, "extra", d))
    m = pd.read_csv(os.path.join(CORPUS, "manifest.csv"))
    base = [p.replace("\\", "/") for p in sorted(glob.glob(os.path.join(V2, "train", "images", "*")))]

    neg, box = [], []
    for r in m[m.role == "negative"].itertuples():
        p = put("neg_" + r.name, os.path.join(PROJECT_DIR, r.path), "")
        if p:
            neg.append(p)
    counts = {}
    for r in m[m.role == "box_train"].itertuples():
        lab = os.path.join(CORPUS, "outlines", r.name + ".txt")
        if not os.path.isfile(lab):
            continue
        with open(lab, encoding="utf-8") as f:
            p = put("box_" + r.name, os.path.join(PROJECT_DIR, r.path), f.read())
        if p:
            box.append(p)
            key = r.source if r.source != "rdd" else f"rdd_{r.country}"
            counts[key] = counts.get(key, 0) + 1

    for name, files in (("neg", base + neg), ("full", base + neg + box)):
        with open(os.path.join(OUT, f"train_{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(files) + "\n")
        with open(os.path.join(OUT, f"data_{name}.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {ARCHIVE.replace(chr(92), '/')}\ntrain: yolo_seg_v3/train_{name}.txt\n"
                    f"val: yolo_seg_v2/val/images\nnames:\n  0: pothole\n")
    print(f"v2 training images: {len(base)}")
    print(f"negatives added:    {len(neg)}")
    print(f"box frames added:   {len(box)}  {counts}")
    print(f"wrote {os.path.relpath(OUT, PROJECT_DIR)}: data_neg.yaml ({len(base) + len(neg)} images), data_full.yaml ({len(base) + len(neg) + len(box)} images)")


if __name__ == "__main__":
    sys.exit(main())
