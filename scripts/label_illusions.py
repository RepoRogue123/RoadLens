"""
Hand-label potholes as real craters vs texture illusions.

This produces the calibration set Phase 3 needs. The semantic override currently
uses a hand-set variance-ratio threshold of 0.9, which diagnostics showed is not
calibrated (the same pothole under three augmentations straddles it). Fitting a
threshold requires labelled examples, and none exist in the project.

SAMPLING DESIGN — why there are two strata
-------------------------------------------
Selecting candidates using the same statistic we intend to calibrate would bias
the fitted threshold. So candidates come from two clearly-tagged pools:

  random    drawn uniformly; preserves the natural class prior, so precision and
            recall computed on this stratum alone are honest.
  enriched  deliberately surfaced dark / low-texture interiors, which is where
            illusions live. Genuine illusions are rare, so a purely random
            sample would contain almost none and teach the threshold nothing.
            Use these to populate the decision boundary, not to estimate priors.

The stratum is recorded per row so the analysis can weight or split accordingly.
Enrichment uses interior Laplacian variance and brightness — cheap image
statistics, deliberately NOT the DINOv2 embedding variance being calibrated.

CONTROLS
--------
  1  real crater          the interior is genuinely broken/rubbled/cavitied
  2  texture illusion     flat road that merely looks dark or discoloured
  3  unclear              cannot tell — excluded from fitting, but recorded
  s  skip (decide later)     u  undo last     q  save and quit

Usage:
    python scripts/label_illusions.py                 # default 120 candidates
    python scripts/label_illusions.py --limit 60
    python scripts/label_illusions.py --no-enrich     # random only
"""
import argparse
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks  # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "illusion_labels.csv")
CANDIDATES_CSV = os.path.join(PROJECT_DIR, "data", "illusion_candidates.csv")

IMAGE_DIRS = [
    "data1/train/images",
    "merged_dataset/train/images",
]

LABEL_KEYS = {
    ord("1"): "real_crater",
    ord("2"): "texture_illusion",
    ord("3"): "unclear",
}

FIELDS = ["image", "pothole_idx", "label", "stratum", "interior_lapvar", "interior_mean"]


# ── interior statistics (cheap, independent of DINOv2) ────────────────────
def interior_stats(image_bgr, mask):
    """Laplacian variance and mean brightness inside the mask."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    inside = mask > 0
    if inside.sum() < 40:
        return None
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap[inside].var()), float(gray[inside].mean())


# ── persistence ───────────────────────────────────────────────────────────
def load_done():
    """Already-labelled (image, idx) pairs, so the tool is resumable."""
    if not os.path.isfile(LABELS_CSV):
        return set(), []
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["image"], r["pothole_idx"]) for r in rows}, rows


def append_row(row):
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    new = not os.path.isfile(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def rewrite(rows):
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ── candidate selection ───────────────────────────────────────────────────
def build_candidates(limit, enrich, seed=42):
    """
    Returns [(path, stratum)]. Cached so the set is stable across sittings —
    re-deciding the sample each run would quietly change the population.
    """
    if os.path.isfile(CANDIDATES_CSV):
        with open(CANDIDATES_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"Using cached candidate list ({len(rows)} images) from {CANDIDATES_CSV}")
        return [(r["path"], r["stratum"]) for r in rows]

    paths = []
    for d in IMAGE_DIRS:
        paths += sorted(glob.glob(os.path.join(PROJECT_DIR, d, "*.jpg")))
    if not paths:
        print("No images found.")
        return []

    rng = random.Random(seed)
    rng.shuffle(paths)

    n_random = limit if not enrich else max(1, int(limit * 0.5))
    chosen = [(p, "random") for p in paths[:n_random]]

    if enrich:
        pool = paths[n_random : n_random + 400]
        print(f"Scanning {len(pool)} images for dark / smooth interiors to enrich…")
        scored = []
        for i, p in enumerate(pool, 1):
            if i % 50 == 0:
                print(f"  {i}/{len(pool)}")
            try:
                masks = get_all_masks(p)
                if not masks:
                    continue
                img = cv2.imread(p)
                st = interior_stats(img, masks[0])
                if st:
                    lapvar, mean = st
                    # illusion-like = smooth interior, and darker rather than brighter
                    scored.append((lapvar + mean * 0.5, p))
            except Exception:
                continue
        scored.sort(key=lambda t: t[0])
        chosen += [(p, "enriched") for _, p in scored[: limit - n_random]]

    os.makedirs(os.path.dirname(CANDIDATES_CSV), exist_ok=True)
    with open(CANDIDATES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "stratum"])
        w.writeheader()
        for p, s in chosen:
            w.writerow({"path": p, "stratum": s})
    print(f"Built candidate list: {len(chosen)} images -> {CANDIDATES_CSV}")
    return chosen


# ── display ───────────────────────────────────────────────────────────────
def build_view(image_bgr, mask, caption, sub):
    """Full frame with outline, beside a zoomed crop of the pothole interior."""
    disp = image_bgr.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(disp, cnts, -1, (0, 255, 255), 2)

    ys, xs = np.where(mask > 0)
    pad = 30
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad)
    crop = image_bgr[y0:y1, x0:x1]

    H = 520
    def fit(im):
        h, w = im.shape[:2]
        return cv2.resize(im, (max(1, int(w * H / h)), H))

    left, right = fit(disp), fit(crop)
    canvas = np.hstack([left, right])

    bar = np.zeros((84, canvas.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, sub, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)
    cv2.putText(bar, "1=real crater   2=texture illusion   3=unclear   s=skip  u=undo  q=quit",
                (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 235, 235), 1)
    return np.vstack([canvas, bar])


def summarise():
    if not os.path.isfile(LABELS_CSV):
        print("No labels yet.")
        return
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"\n{len(rows)} labels in {os.path.relpath(LABELS_CSV, PROJECT_DIR)}")
    for stratum in ("random", "enriched"):
        sub = [r for r in rows if r["stratum"] == stratum]
        if not sub:
            continue
        counts = {}
        for r in sub:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print(f"  {stratum:<9} n={len(sub):<4} " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    n_illusion = sum(1 for r in rows if r["label"] == "texture_illusion")
    n_real = sum(1 for r in rows if r["label"] == "real_crater")
    print(f"\n  usable for fitting: {n_real} real_crater / {n_illusion} texture_illusion")
    if n_illusion < 15:
        print("  ⚠  Fewer than ~15 illusions — the fitted threshold will be weak.")
        print("     Re-run with --enrich to surface more candidates.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--no-enrich", action="store_true")
    ap.add_argument("--summary", action="store_true", help="print label stats and exit")
    args = ap.parse_args()

    if args.summary:
        summarise()
        return

    candidates = build_candidates(args.limit, enrich=not args.no_enrich)
    if not candidates:
        return

    done, _ = load_done()
    todo = [(p, s) for p, s in candidates if (os.path.basename(p), "0") not in done]
    print(f"\n{len(done)} already labelled · {len(todo)} remaining")
    print("Judge the ZOOMED panel: broken asphalt / rubble / cavity = real crater;")
    print("flat but dark or discoloured = texture illusion.\n")

    session = []
    for i, (path, stratum) in enumerate(todo, 1):
        try:
            masks = get_all_masks(path)
        except Exception as e:
            print(f"  skip (segmentation failed): {os.path.basename(path)} — {e}")
            continue
        if not masks:
            continue
        mask = masks[0]
        img = cv2.imread(path)
        if img is None:
            continue
        st = interior_stats(img, mask)
        if st is None:
            continue
        lapvar, mean = st

        view = build_view(
            img, mask,
            f"[{i}/{len(todo)}]  {os.path.basename(path)[:52]}",
            f"stratum={stratum}   interior lap-var={lapvar:.0f}   mean brightness={mean:.0f}",
        )
        cv2.imshow("label illusions", view)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k == ord("q"):
                cv2.destroyAllWindows()
                summarise()
                return
            if k == ord("s"):
                break
            if k == ord("u"):
                if session:
                    _, rows = load_done()
                    if rows:
                        removed = rows.pop()
                        rewrite(rows)
                        session.pop()
                        print(f"  undid: {removed['image']} ({removed['label']})")
                else:
                    print("  nothing to undo in this session")
                continue
            if k in LABEL_KEYS:
                row = {
                    "image": os.path.basename(path),
                    "pothole_idx": "0",
                    "label": LABEL_KEYS[k],
                    "stratum": stratum,
                    "interior_lapvar": f"{lapvar:.2f}",
                    "interior_mean": f"{mean:.2f}",
                }
                append_row(row)
                session.append(row)
                print(f"  {LABEL_KEYS[k]:<17} {os.path.basename(path)[:48]}")
                break

    cv2.destroyAllWindows()
    summarise()


if __name__ == "__main__":
    main()
