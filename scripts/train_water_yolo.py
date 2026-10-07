"""
Train a learned water segmenter (YOLOv8n-seg, one class: puddle) on HanYang.

Roadmap item M3.3. The hand-built water detector combines optical cues, CLIPSeg and
a fitted logistic model; the open question (M3.4) is whether a segmenter trained
directly on puddle outlines does better. This trains it; the comparison is
`scripts/eval_water_puddle1000.py --segmenter yolo-segmentation/model/water_best.pt`.

Splits
------
    train   HanYang train, minus 10% of photographs
    val     that 10%, grouped by source_stem so Roboflow augmentations of one
            photograph never straddle train and val — used only for early stopping
    test    HanYang valid (300 images) — the same held-out set the logistic water
            model was scored on (regions_HELD_OUT_SEMONLY.csv). Never used here.

No images are copied: Ultralytics reads the split lists and finds each label by
swapping /images/ for /labels/.

Usage:
    python scripts/train_water_yolo.py [--epochs 100]
    python scripts/train_water_yolo.py --resume      # continue an interrupted run from last.pt
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from dataset_registry import source_stem          # noqa: E402

DATA = os.path.join(PROJECT_DIR, "puddle segmentation.v8i.yolov8")
WORK = os.path.join(PROJECT_DIR, "archive", "water_seg")
RUNS = os.path.join(PROJECT_DIR, "archive", "yolo_runs")
OUT = os.path.join(PROJECT_DIR, "yolo-segmentation", "model", "water_best.pt")
SEED = 20260929


def write_splits():
    imgs = sorted(glob.glob(os.path.join(DATA, "train", "images", "*")))
    stems = sorted({source_stem(os.path.basename(p)) for p in imgs})
    val_stems = set(np.random.default_rng(SEED).permutation(stems)[: len(stems) // 10])
    os.makedirs(WORK, exist_ok=True)
    tr = [p for p in imgs if source_stem(os.path.basename(p)) not in val_stems]
    va = [p for p in imgs if source_stem(os.path.basename(p)) in val_stems]
    for name, paths in (("train", tr), ("val", va)):
        with open(os.path.join(WORK, f"{name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(p.replace("\\", "/") for p in paths) + "\n")
    test = os.path.join(DATA, "valid", "images").replace("\\", "/")
    yaml = os.path.join(WORK, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"train: {WORK.replace(chr(92), '/')}/train.txt\n"
                f"val: {WORK.replace(chr(92), '/')}/val.txt\n"
                f"test: {test}\nnames:\n  0: puddle\n")
    print(f"train {len(tr)} images, val {len(va)} ({len(val_stems)} of {len(stems)} photographs)")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO
    last = os.path.join(RUNS, "water_seg", "weights", "last.pt")
    if args.resume:
        YOLO(last).train(resume=True)       # same splits, epochs and seed as the original run
    else:
        _train_fresh(args)
    best = os.path.join(RUNS, "water_seg", "weights", "best.pt")
    shutil.copy2(best, OUT)
    print(f"copied {best} -> {os.path.relpath(OUT, PROJECT_DIR)}")


def _train_fresh(args):
    from ultralytics import YOLO
    yaml = write_splits()
    model = YOLO("yolov8n-seg.pt")          # COCO-pretrained; water is not a pothole class
    model.train(data=yaml, epochs=args.epochs, patience=20, imgsz=640, batch=args.batch,
                seed=SEED, deterministic=True, workers=4,
                project=RUNS, name="water_seg", exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
