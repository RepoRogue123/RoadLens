"""
How does a segmenter behave on dashcam frames from a country it never trained on?

Uses the RDD country held out by scripts/build_corpus_manifest.py. RDD has boxes, not
outlines, so the questions are the ones boxes can answer, at the production settings
(segmentation._extract_binary_masks):

  On frames WITH a pothole box
    box recall    share of boxes that some predicted mask sits in: at least half of the
                  mask's pixels inside the box (loosened 10%), covering at least 10% of it
    stray/img     predicted masks with under 10% of their pixels in any box

  On frames WITHOUT a pothole box (cracks only, or clean road)
    frames hit    share of frames where the segmenter outlined something
    masks/img     how many
    top third     share of those masks centred in the top third of the frame: sky, trees,
                  buildings. The failure seen on RDD with segmenter v2.

Caveat: RDD's D40 class can be looser than "pothole" (in the first Japanese release,
RDD2018, it also covered rutting, bumps and separation; check the contact sheet from
scripts/boxes_to_polygons.py --sheet), so box recall is a relative measure between
segmenters, not an absolute one.

Usage:
    python scripts/eval_yolo_rdd.py yolo-segmentation/model/best_v2.pt@0.35 other.pt@0.35
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import _extract_binary_masks                     # noqa: E402

MANIFEST = os.path.join(PROJECT_DIR, "archive", "corpus_v3", "manifest.csv")
OUT = os.path.join(PROJECT_DIR, "ml_results", "yolo_seg_rdd_heldout.json")
LOOSEN = 0.10


def score(model, conf, m):
    found = n_box = stray = n_pos = 0
    for r in m[m.role == "box_test"].itertuples():
        img = cv2.imread(os.path.join(PROJECT_DIR, r.path))
        if img is None:
            continue
        h, w = img.shape[:2]
        preds = _extract_binary_masks(img, model, conf_threshold=conf)
        areas = [float(p.sum()) for p in preds]
        inside_any = np.zeros(len(preds))
        n_pos += 1
        for x0, y0, x1, y1 in json.loads(r.boxes):
            dx, dy = (x1 - x0) * LOOSEN / 2, (y1 - y0) * LOOSEN / 2
            a, b, c, d = int(max(0, x0 - dx)), int(max(0, y0 - dy)), int(min(w, x1 + dx)), int(min(h, y1 + dy))
            box_area = max((c - a) * (d - b), 1)
            hit = False
            for i, p in enumerate(preds):
                inside = float(p[b:d, a:c].sum())
                inside_any[i] = max(inside_any[i], inside / max(areas[i], 1))
                hit = hit or (inside >= 0.5 * areas[i] and inside >= 0.1 * box_area)
            found += hit
            n_box += 1
        stray += int((inside_any < 0.1).sum())
    hit_frames = n_masks = top = n_neg = 0
    for r in m[m.role == "neg_test"].itertuples():
        img = cv2.imread(os.path.join(PROJECT_DIR, r.path))
        if img is None:
            continue
        preds = _extract_binary_masks(img, model, conf_threshold=conf)
        n_neg += 1
        hit_frames += bool(preds)
        n_masks += len(preds)
        top += sum(np.nonzero(p)[0].mean() < img.shape[0] / 3 for p in preds)
    return {"box_recall": found / max(n_box, 1), "n_boxes": n_box, "stray_per_img": stray / max(n_pos, 1),
            "neg_frames_hit": hit_frames / max(n_neg, 1), "neg_masks_per_img": n_masks / max(n_neg, 1),
            "neg_masks_top_third": int(top), "neg_masks": int(n_masks), "n_neg_frames": n_neg}


def main():
    m = pd.read_csv(MANIFEST)
    country = m[m.role == "box_test"].country.iloc[0]
    results = {}
    for arg in sys.argv[1:]:
        wpath, _, conf = arg.partition("@")
        conf = float(conf) if conf else 0.25
        results[f"{os.path.basename(wpath)}@{conf:g}"] = score(YOLO(wpath), conf, m)
    print(f"\nheld-out country: {country}")
    print(f"{'weights':22s} {'box recall':>11s} {'stray/img':>10s} | {'clean frames hit':>17s} {'masks/img':>10s} {'in top third':>13s}")
    for name, r in results.items():
        print(f"{name:22s} {r['box_recall']:11.1%} {r['stray_per_img']:10.2f} | {r['neg_frames_hit']:17.1%} "
              f"{r['neg_masks_per_img']:10.2f} {r['neg_masks_top_third']:6d}/{r['neg_masks']:<6d}   "
              f"(n={r['n_boxes']} boxes, {r['n_neg_frames']} frames without a pothole)")
    prev = {}
    if os.path.isfile(OUT):
        with open(OUT, encoding="utf-8") as f:
            prev = json.load(f)
    prev.update(results)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
