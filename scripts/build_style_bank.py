"""
Style bank: how photos from every other dataset are lit and coloured, for depth training.

PothRGBD is the only source of millimetres, and it is one camera on one kind of road.
Fourier domain adaptation (Yang and Soatto, CVPR 2020) lets the other datasets contribute
without any labels: the low-frequency AMPLITUDE of an image's spectrum holds its overall
colour, exposure and illumination; the phase holds the scene. Swapping in another photo's
low-frequency amplitude re-lights a PothRGBD frame like that photo while every edge,
and so every depth label, stays where it was.

Only that small block of the spectrum is needed, so the bank stores a 41x41x3 block per
photo (18 KB) rather than the photo: 4,000 photos in about 80 MB.

Sources (training portions only; nothing from any evaluation split):

    kaggle      data1/train                     phone and web photos
    pothole600  pothole600/training             stereo rig close-ups
    mendeley    Mendeley IMG                    UK phone photos
    rdd_<c>     rdd_temp/<country>/train        dashcams, six countries (drone set left out)
    hanyang     puddle segmentation train       wet Korean city roads
    potholedepth_phone  archive/potholedepth    phone video of streets in Lahore

Output: archive/style_bank/amp.npy, sources.csv

Usage:
    python scripts/build_style_bank.py [--per-country 400]
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from dataset_registry import source_stem                                        # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "archive", "style_bank")
HALF = 20                                    # block is (2*HALF+1)^2; training uses at most +-6
SEED = 20261003
RDD_COUNTRIES = ("Japan", "India", "Czech", "Norway", "United_States", "China_MotorBike")
MENDELEY = os.path.join(PROJECT_DIR, "mendeley_water_filled_dataset",
                        "An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications", "IMG")


def one_per_photo(paths):
    """Roboflow exports hold several augmented copies of each photo; keep one."""
    seen, out = set(), []
    for p in sorted(paths):
        s = source_stem(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def amplitude_block(path):
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2 if os.path.getsize(path) > 1_500_000 else cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (R.IN_W, R.IN_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    amp = np.fft.fftshift(np.abs(np.fft.fft2(img, axes=(0, 1))), axes=(0, 1))
    ch, cw = R.IN_H // 2, R.IN_W // 2
    return amp[ch - HALF:ch + HALF + 1, cw - HALF:cw + HALF + 1].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-country", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    img = lambda d: [p for p in glob.glob(os.path.join(PROJECT_DIR, d, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))]   # noqa: E731

    pools = {
        "kaggle": one_per_photo(img("data1/train/images")),
        "pothole600": img("pothole600/training/rgb"),
        "mendeley": sorted(glob.glob(os.path.join(MENDELEY, "*.jpg"))),
        "hanyang": one_per_photo(img("puddle segmentation.v8i.yolov8/train/images"))[:400],
        # ~14 phone video clips of Lahore streets; every 60th frame, so near-identical
        # neighbours are not all taken. (Its depth files are unusable: scripts/depth_sources.py.)
        "potholedepth_phone": sorted(glob.glob(os.path.join(PROJECT_DIR, "archive", "potholedepth", "images", "*.jpg")))[::60],
    }
    for c in RDD_COUNTRIES:
        files = sorted(glob.glob(os.path.join(PROJECT_DIR, "rdd_temp", c, c, "train", "images", "*.jpg")))
        pools[f"rdd_{c}"] = list(rng.choice(files, min(args.per_country, len(files)), replace=False))

    blocks, rows = [], []
    for name, files in pools.items():
        n = 0
        for p in files:
            b = amplitude_block(p)
            if b is None:
                continue
            blocks.append(b)
            rows.append({"source": name, "path": os.path.relpath(p, PROJECT_DIR).replace("\\", "/")})
            n += 1
        print(f"  {name:20s} {n:5d} photos", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "amp.npy"), np.stack(blocks))
    with open(os.path.join(OUT_DIR, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote archive/style_bank/amp.npy: {len(blocks)} photos, {np.stack(blocks).nbytes / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
