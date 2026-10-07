"""
Hand-label potholes as holding water or dry.

Why this exists
---------------
Two water detectors disagree and no dataset can settle it. The learned segmenter
beats the cue ensemble on HanYang (missed water 33 vs 90 of 442), but HanYang
regions are *water outlines*; in the pipeline the region is the *pothole outline*
and water often fills only its centre. No dataset here labels pothole water — the
Mendeley "Water-Filled and Dry Potholes" set labels only `pothole` boxes. So
~150 potholes need a human verdict, then scripts/eval_pothole_water.py decides
the detector and the coverage threshold.

Sampling
--------
Random with a fixed seed, never selected by either detector's output (that would
bias the comparison). Three sources, so the verdict is not one camera's:
    mendeley   UK phone photos, advertised as water-filled and dry  (60)
    kaggle     mixed web photos                                     (45)
    pothrgbd   RealSense close-ups, mostly dry                      (45)
Up to two potholes per photo, outlined by the production segmenter. The
candidate list is cached so the population does not change between sittings.

CONTROLS
--------
  w  holds water   (any standing water inside the outline, even a shallow film)
  d  dry           (damp-looking soil or dark asphalt counts as dry)
  3  unclear       recorded, excluded from evaluation
  s  skip (decide later)     u  undo last     q  save and quit

Usage:
    python scripts/label_pothole_water.py
"""
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks                        # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR            # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "pothole_water_labels.csv")
CANDIDATES_CSV = os.path.join(PROJECT_DIR, "data", "pothole_water_candidates.csv")
MENDELEY = os.path.join(PROJECT_DIR, "mendeley_water_filled_dataset",
                        "An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications", "IMG")
SOURCES = {
    "mendeley": (os.path.join(MENDELEY, "*.jpg"), 60),
    "kaggle": (os.path.join(PROJECT_DIR, "data1", "*", "images", "*.jpg"), 45),
    "pothrgbd": (os.path.join(DATA_DIR, "images", "*.jpg"), 45),
}
KEYS = {ord("w"): "water", ord("d"): "dry", ord("3"): "unclear"}
FIELDS = ["image", "pothole_idx", "source", "label"]
SEED = 20260929


def build_candidates():
    """Random photos per source until each quota of potholes is met; cached."""
    if os.path.isfile(CANDIDATES_CSV):
        with open(CANDIDATES_CSV, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    rng = random.Random(SEED)
    rows = []
    for source, (pattern, quota) in SOURCES.items():
        paths = sorted(glob.glob(pattern))
        rng.shuffle(paths)
        n = 0
        for p in paths:
            if n >= quota:
                break
            for i in range(min(2, len(get_all_masks(p)))):
                if n >= quota:
                    break
                rows.append({"image": os.path.relpath(p, PROJECT_DIR), "pothole_idx": i, "source": source})
                n += 1
        print(f"  {source}: {n} potholes")
    rng.shuffle(rows)                      # interleave sources so fatigue is not source-specific
    os.makedirs(os.path.dirname(CANDIDATES_CSV), exist_ok=True)
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "pothole_idx", "source"])
        w.writeheader()
        w.writerows(rows)
    return rows


def load_done():
    if not os.path.isfile(LABELS_CSV):
        return []
    with open(LABELS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save(rows):
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def build_view(image_bgr, mask, caption):
    """Full frame with the outline, beside a zoomed crop of the pothole."""
    disp = image_bgr.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, cnts, -1, (0, 255, 255), 2)
    ys, xs = np.where(mask > 0)
    pad = 30
    crop = image_bgr[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]

    def fit(im, h=520):
        return cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))

    canvas = np.hstack([fit(disp), fit(crop)])
    bar = np.zeros((60, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, "w=holds water   d=dry   3=unclear   s=skip  u=undo  q=quit",
                (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 235, 235), 1)
    return np.vstack([canvas, bar])


def main():
    cands = build_candidates()
    done = load_done()
    seen = {(r["image"], str(r["pothole_idx"])) for r in done}
    todo = [c for c in cands if (c["image"], str(c["pothole_idx"])) not in seen]
    print(f"{len(done)} labelled, {len(todo)} to go")
    win = "RoadLens - does this pothole hold water?"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    while i < len(todo):
        c = todo[i]
        path = os.path.join(PROJECT_DIR, c["image"])
        img = cv2.imread(path)
        masks = get_all_masks(path)
        idx = int(c["pothole_idx"])
        if img is None or idx >= len(masks):
            i += 1
            continue
        caption = f"[{len(done) + 1}/{len(cands)}] {c['source']}  {os.path.basename(path)}  pothole {idx + 1}"
        cv2.imshow(win, build_view(img, masks[idx], caption))
        k = cv2.waitKey(0) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
            continue
        if k == ord("u") and done:
            done.pop()
            save(done)
            i = max(0, i - 1)
            continue
        if k in KEYS:
            done.append({"image": c["image"], "pothole_idx": idx, "source": c["source"], "label": KEYS[k]})
            save(done)
            i += 1
    cv2.destroyAllWindows()
    counts = {}
    for r in done:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"saved {len(done)} labels to {os.path.relpath(LABELS_CSV, PROJECT_DIR)}: {counts}")
    if counts.get("water", 0) < 20:
        print("  fewer than 20 water labels — the evaluation will be too noisy to choose a threshold")


if __name__ == "__main__":
    main()
