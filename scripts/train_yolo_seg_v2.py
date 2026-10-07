"""
Fine-tune the production YOLOv8n-seg on the v2 outline set (build_yolo_seg_v2.py).

Starts from the production weights rather than COCO, so the Kaggle knowledge is kept
and the extra sources add to it. Same architecture and input size as production, so
a swap changes nothing downstream except the masks.

The result is written to yolo-segmentation/model/best_v2.pt. It does NOT replace
best.pt: adoption is decided by scripts/eval_yolo_seg.py on the held-out test sets.

Usage:
    python scripts/train_yolo_seg_v2.py [--epochs 100]
"""
import argparse
import os
import shutil

from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2", "data.yaml")
BASE = os.path.join(PROJECT_DIR, "yolo-segmentation", "model", "best.pt")
RUNS = os.path.join(PROJECT_DIR, "archive", "yolo_runs")
OUT = os.path.join(PROJECT_DIR, "yolo-segmentation", "model", "best_v2.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    # v3 (scripts/build_yolo_seg_v3.py) reuses this script: same base, same recipe, other data.
    ap.add_argument("--data", default=DATA, help="data yaml (default: the v2 set)")
    ap.add_argument("--name", default="seg_v2", help="run folder under archive/yolo_runs/")
    ap.add_argument("--out", default=OUT, help="where the best weights are copied")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run of this name")
    args = ap.parse_args()

    last = os.path.join(RUNS, args.name, "weights", "last.pt")
    if args.resume and os.path.isfile(last):
        YOLO(last).train(resume=True)
    else:
        YOLO(BASE).train(data=args.data, epochs=args.epochs, patience=20, imgsz=640, batch=args.batch,
                         seed=20260925, deterministic=True, workers=4,
                         project=RUNS, name=args.name, exist_ok=True, plots=True)
    best = os.path.join(RUNS, args.name, "weights", "best.pt")
    shutil.copy2(best, args.out)
    print(f"copied {best} -> {os.path.relpath(args.out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
