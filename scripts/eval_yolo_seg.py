"""
Compare segmentation weights on the held-out v2 test sets.

Two kinds of number, because they answer different questions:

1. Ultralytics mask mAP50 / mAP50-95 — the standard benchmark, over all confidences.
2. What the pipeline actually experiences, at the production settings
   (conf 0.25, min_area 100, segmentation._extract_binary_masks):
     found@0.5    share of labelled potholes matched by a mask with IoU >= 0.5
     found@0.1    share matched at all (the rule the metric trainset used)
     mean IoU     best IoU per labelled pothole, 0 when missed
     false/img    predicted masks overlapping no labelled pothole (IoU < 0.1), per image

Each argument is a weights path, optionally with a confidence threshold after '@'
(default 0.25, the production setting). Choose thresholds on the validation split, not here.

Usage:
    python scripts/eval_yolo_seg.py yolo-segmentation/model/best.pt yolo-segmentation/model/best_v2.pt@0.35
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import _extract_binary_masks                     # noqa: E402

ROOT = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
SETS = ("test_kaggle", "test_p600", "test_pothrgbd")


def gt_masks(label_path, h, w):
    out = []
    with open(label_path, encoding="utf-8") as f:
        for ln in f:
            v = ln.split()
            if len(v) < 7:
                continue
            pts = (np.array(v[1:], dtype=np.float64).reshape(-1, 2) * [w, h]).astype(np.int32)
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [pts], 1)
            out.append(m)
    return out


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def pipeline_scores(model, split, conf=0.25):
    best, false_pos, n_img = [], 0, 0
    for ip in sorted(glob.glob(os.path.join(ROOT, split, "images", "*"))):
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        lp = os.path.join(ROOT, split, "labels", os.path.splitext(os.path.basename(ip))[0] + ".txt")
        gts = gt_masks(lp, h, w)
        preds = _extract_binary_masks(img, model, conf_threshold=conf)
        n_img += 1
        for g in gts:
            best.append(max((iou(p, g) for p in preds), default=0.0))
        for p in preds:
            if max((iou(p, g) for g in gts), default=0.0) < 0.1:
                false_pos += 1
    best = np.array(best)
    return {"found@0.5": float((best >= 0.5).mean()), "found@0.1": float((best >= 0.1).mean()),
            "mean_iou": float(best.mean()), "false_per_img": false_pos / max(n_img, 1),
            "n_potholes": int(len(best)), "n_images": n_img}


def main():
    # "--tag=NAME" writes ml_results/yolo_seg_NAME_eval.json instead of overwriting the v2 record
    tag = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tag=")), "v2")
    weights = [a for a in sys.argv[1:] if not a.startswith("--tag=")] \
        or [os.path.join(PROJECT_DIR, "yolo-segmentation", "model", "best.pt")]
    results = {}
    for arg in weights:
        wpath, _, conf = arg.partition("@")
        conf = float(conf) if conf else 0.25
        name = f"{os.path.basename(wpath)}@{conf:g}"
        model = YOLO(wpath)
        results[name] = {}
        for split in SETS:
            m = model.val(data=os.path.join(ROOT, f"{split}.yaml"), split="val", imgsz=640,
                          batch=8, plots=False, verbose=False, project=os.path.join(ROOT, "_val"),
                          name=f"{name}_{split}", exist_ok=True)
            r = {"mask_mAP50": float(m.seg.map50), "mask_mAP50_95": float(m.seg.map)}
            r.update(pipeline_scores(model, split, conf))
            results[name][split] = r

    print(f"\n{'weights':18s} {'test set':15s} {'mAP50':>6s} {'mAP50-95':>9s} {'found@0.5':>10s} "
          f"{'found@0.1':>10s} {'mean IoU':>9s} {'false/img':>10s}")
    for name, per in results.items():
        for split, r in per.items():
            print(f"{name:18s} {split:15s} {r['mask_mAP50']:6.3f} {r['mask_mAP50_95']:9.3f} "
                  f"{r['found@0.5']:10.3f} {r['found@0.1']:10.3f} {r['mean_iou']:9.3f} {r['false_per_img']:10.2f}"
                  f"   (n={r['n_potholes']})")
    out = os.path.join(PROJECT_DIR, "ml_results", f"yolo_seg_{tag}_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
