"""
Pothole close-ups from the datasets that have no depth, for the consistency loss.

The relief network only ever sees PothRGBD's camera. These crops let it see potholes from
phones and dashcams in six countries during training — with no depth label, so the only
thing asked of it is that its answer does not change when the photo is mirrored and
re-lit (scripts/train_relief_depth.py --consistency).

Each crop is a 4:3 window around one labelled pothole, three times the pothole's size, so
the pothole sits in the frame roughly as it does in a PothRGBD close-up. Sources, training
portions only (scripts/build_corpus_manifest.py decides which frames those are):

    rdd       box_train frames (India, Czech, United States, China; Norway's potholes are
              a few pixels wide and are skipped by the size floor)
    mendeley  box_train frames
    kaggle    data1/train, one copy per photograph, box taken around each outline

Output: archive/corpus_v3/unlabelled_crops/*.jpg (518x392), crops.csv

Usage:
    python scripts/build_unlabelled_crops.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from dataset_registry import source_stem                                        # noqa: E402

CORPUS = os.path.join(PROJECT_DIR, "archive", "corpus_v3")
OUT = os.path.join(CORPUS, "unlabelled_crops")
CONTEXT = 3.0                 # window width as a multiple of the pothole's larger side
MIN_BOX_PX = 40               # a pothole smaller than this is too few pixels to say anything about
MIN_WINDOW_PX = 200


def window(box, w, h):
    """4:3 window around a box, clipped to the image; None if the pothole is too small."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    if side < MIN_BOX_PX:
        return None
    ww = max(side * CONTEXT, MIN_WINDOW_PX)
    wh = ww * R.IN_H / R.IN_W
    if ww > w or wh > h:                                   # the pothole fills the photo: use the largest 4:3 that fits
        ww = min(w, h * R.IN_W / R.IN_H)
        wh = ww * R.IN_H / R.IN_W
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a = int(np.clip(cx - ww / 2, 0, w - ww))
    b = int(np.clip(cy - wh / 2, 0, h - wh))
    return a, b, int(ww), int(wh)


def kaggle_frames():
    seen = set()
    for p in sorted(glob.glob(os.path.join(PROJECT_DIR, "data1", "train", "images", "*"))):
        stem = source_stem(p)
        if stem in seen:
            continue
        seen.add(stem)
        img = cv2.imread(p)
        lab = os.path.join(PROJECT_DIR, "data1", "train", "labels", os.path.splitext(os.path.basename(p))[0] + ".txt")
        if img is None or not os.path.isfile(lab):
            continue
        h, w = img.shape[:2]
        boxes = []
        with open(lab, encoding="utf-8") as f:
            for ln in f:
                v = ln.split()
                if len(v) >= 7:
                    pts = np.array(v[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
                    boxes.append([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
        yield "kaggle", "kaggle_" + stem, img, boxes


def manifest_frames():
    m = pd.read_csv(os.path.join(CORPUS, "manifest.csv"))
    for r in m[m.role == "box_train"].itertuples():
        img = cv2.imread(os.path.join(PROJECT_DIR, r.path))
        if img is not None:
            yield (r.source if r.source != "rdd" else f"rdd_{r.country}"), r.name, img, json.loads(r.boxes)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for frames in (kaggle_frames(), manifest_frames()):
        for source, name, img, boxes in frames:
            h, w = img.shape[:2]
            for i, b in enumerate(boxes[:3]):
                win = window(b, w, h)
                if win is None:
                    continue
                a, c, ww, wh = win
                crop = cv2.resize(img[c:c + wh, a:a + ww], (R.IN_W, R.IN_H), interpolation=cv2.INTER_AREA)
                fn = f"{name}_{i}.jpg"
                cv2.imwrite(os.path.join(OUT, fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"source": source, "file": fn, "window_px": ww})
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(CORPUS, "crops.csv"), index=False)
    print(d.groupby("source").size().to_string())
    print(f"wrote {len(d)} crops to {os.path.relpath(OUT, PROJECT_DIR)}; "
          f"{(d.window_px < R.IN_W).mean():.0%} were enlarged to reach 518 px")


if __name__ == "__main__":
    main()
