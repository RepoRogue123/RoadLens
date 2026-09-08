"""
Evaluate the Phase 4 water ensemble against a labelled water set.

Why this exists
---------------
water_detection.py was tuned against four hand-picked images. Its own docstring
records the provenance:

    "pic-52 was falsely classified DRY (16.9%) but has subtle moisture"
    "pic-65 was falsely classified WATER (47.1%) but is dry gravel in sunlight"
    "This alone fixes pic-65 false positive"

Constants fitted to two named images are a memory of those two images. The
module has never been given a precision or recall figure. This script supplies
one, and can refit the weights from labels instead of from judgement.

Primary target is Puddle-1000 (985 road-ponding images: 357 structured "ONR" +
628 unstructured "OFR"), plus its Foggy and Night variants. Links are in the
AGSENet repo README: https://github.com/Lyu-Dakang/AGSENet
It works with any dataset of images plus binary water masks.

What is actually being measured
-------------------------------
This is REGION CLASSIFICATION, not segmentation. detect_water answers "is there
water inside this region", so the evaluation gives it regions and checks the
answer. It does NOT produce a segmentation IoU, and its numbers are therefore
NOT comparable with the IoU figures reported by AGSENet or ABCDWaveNet. Saying
otherwise would be a category error.

Negatives are SHAPE-MATCHED: each negative reuses a positive region's exact
shape, translated to a water-free location in the same image. Several cues are
size-sensitive, so drawing negatives of arbitrary size would let the classifier
separate them on area rather than on water. This is the same size-confound
control applied to the Phase 3 variance ratio.

Usage
-----
    # evaluate as shipped
    python scripts/eval_water_puddle1000.py --data data/puddle1000/ONR

    # add the two new cues (slower: one ViT + one decomposition per image)
    python scripts/eval_water_puddle1000.py --data data/puddle1000/ONR --semantic --residual

    # measure how much the hand-tuned non-linear corrections account for
    python scripts/eval_water_puddle1000.py --data data/puddle1000/ONR --no-corrections

    # fit cue weights and threshold, write ml_results/phase4_calibration/water_params.json
    python scripts/eval_water_puddle1000.py --data data/puddle1000/ONR --semantic --residual --fit
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import date
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import water_config                                    # noqa: E402
from water_detection import detect_water               # noqa: E402

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "phase4_calibration")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# A region smaller than this is too thin for the cues to say anything useful.
MIN_REGION_PX = 400
# How many placement attempts before giving up on a shape-matched negative.
NEG_PLACEMENT_TRIES = 40

RNG = np.random.default_rng(20260907)


# ── dataset discovery ────────────────────────────────────────────────────

def _stem(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def _yolo_seg_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Roboflow YOLOv8-segmentation layout: <root>/[split/]images + labels, where
    each label is `class x1 y1 x2 y2 ...` with NORMALISED polygon coordinates.

    Returned "mask" paths are .txt files; load_gt_mask rasterises them on demand
    rather than writing several thousand PNGs to disk for a single evaluation.
    """
    roots = [data_dir] + [os.path.join(data_dir, s)
                          for s in ("train", "valid", "val", "test")]
    pairs = []
    for root in roots:
        idir, ldir = os.path.join(root, "images"), os.path.join(root, "labels")
        if not (os.path.isdir(idir) and os.path.isdir(ldir)):
            continue
        labels = {_stem(f): os.path.join(ldir, f)
                  for f in os.listdir(ldir) if f.lower().endswith(".txt")}
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            lp = labels.get(_stem(f))
            if lp:
                pairs.append((os.path.join(idir, f), lp))
    return pairs


def load_gt_mask(path: str, h: int, w: int) -> Optional[np.ndarray]:
    """
    Load ground truth as a binary mask, from either an image or a YOLO-seg .txt.

    Polygons are filled rather than approximated by their bounding box: a box
    around a puddle contains a great deal of dry road, which would contaminate
    both the positive region and the shape-matched negative derived from it.
    """
    if path.lower().endswith(".txt"):
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 7:          # class + at least 3 points
                        continue
                    coords = np.array([float(v) for v in parts[1:]], dtype=np.float32)
                    if coords.size % 2:
                        coords = coords[:-1]
                    pts = coords.reshape(-1, 2) * np.array([w, h], dtype=np.float32)
                    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        except Exception:
            return None
        return mask * 255

    gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        return None
    if gt.shape[:2] != (h, w):
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
    return gt


def discover_pairs(data_dir: str,
                   images_sub: Optional[str],
                   masks_sub: Optional[str]) -> List[Tuple[str, str]]:
    """
    Find (image, ground-truth) pairs. Public water sets are not consistent about
    directory naming, so several common layouts are tried before giving up.
    """
    # YOLO-seg first: `images/` + `labels/` would otherwise be matched by the
    # image-pair branch below, which looks for image files in the label folder
    # and finds none.
    yolo = _yolo_seg_pairs(data_dir)
    if yolo:
        print(f"  layout: YOLO-seg polygons  ({len(yolo)} pairs)")
        return yolo

    candidates = []
    if images_sub and masks_sub:
        candidates.append((images_sub, masks_sub))
    candidates += [
        ("images", "masks"), ("image", "mask"),
        ("images", "labels"), ("img", "gt"),
        ("JPEGImages", "SegmentationClass"),
        ("rgb", "gt"), ("train", "train_labels"),
    ]

    for img_sub, msk_sub in candidates:
        idir = os.path.join(data_dir, img_sub)
        mdir = os.path.join(data_dir, msk_sub)
        if not (os.path.isdir(idir) and os.path.isdir(mdir)):
            continue

        masks_by_stem = {}
        for m in os.listdir(mdir):
            if m.lower().endswith(IMG_EXT):
                masks_by_stem[_stem(m)] = os.path.join(mdir, m)

        pairs = []
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            mp = masks_by_stem.get(_stem(f))
            if mp:
                pairs.append((os.path.join(idir, f), mp))
        if pairs:
            print(f"  layout: {img_sub}/ + {msk_sub}/  ({len(pairs)} pairs)")
            return pairs

    # Flat layout: foo.jpg beside foo_mask.png / foo_gt.png
    pairs = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*"))):
        if not f.lower().endswith(IMG_EXT):
            continue
        s = _stem(f)
        if s.endswith(("_mask", "_gt", "_label")):
            continue
        for suffix in ("_mask", "_gt", "_label"):
            hits = glob.glob(os.path.join(data_dir, f"{s}{suffix}.*"))
            if hits:
                pairs.append((f, hits[0]))
                break
    if pairs:
        print(f"  layout: flat with _mask/_gt suffix  ({len(pairs)} pairs)")
    return pairs


# ── region construction ──────────────────────────────────────────────────

def regions_from_mask(gt: np.ndarray, max_regions: int = 3) -> List[np.ndarray]:
    """Connected components of the ground-truth water mask, largest first."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (gt > 0).astype(np.uint8), connectivity=8
    )
    order = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    out = []
    for i in order[:max_regions]:
        if stats[i, cv2.CC_STAT_AREA] < MIN_REGION_PX:
            continue
        out.append((labels == i).astype(np.uint8))
    return out


def sample_negative(shape_mask: np.ndarray, gt: np.ndarray) -> Optional[np.ndarray]:
    """
    Translate `shape_mask` to a water-free location in the same image.

    Shape-matched by construction: identical area and outline, different place.
    Restricted to the lower two-thirds of the frame, where road actually is,
    and required to miss a dilated version of the ground truth so a negative
    never clips the edge of real water.
    """
    h, w = gt.shape[:2]
    ys, xs = np.nonzero(shape_mask)
    if ys.size == 0:
        return None

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    rh, rw = y1 - y0 + 1, x1 - x0 + 1
    if rh >= h or rw >= w:
        return None

    patch = shape_mask[y0:y1 + 1, x0:x1 + 1]
    forbidden = cv2.dilate((gt > 0).astype(np.uint8), np.ones((15, 15), np.uint8))

    y_lo = max(h // 3, 0)
    for _ in range(NEG_PLACEMENT_TRIES):
        if h - rh <= y_lo or w - rw <= 0:
            break
        ny = int(RNG.integers(y_lo, h - rh))
        nx = int(RNG.integers(0, w - rw))
        cand = np.zeros((h, w), dtype=np.uint8)
        cand[ny:ny + rh, nx:nx + rw] = patch
        if not np.any(cand & forbidden):
            return cand
    return None


# ── metrics ──────────────────────────────────────────────────────────────

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec == prec and rec == rec and prec + rec else float("nan"))
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=prec, recall=rec, f1=f1, accuracy=acc)


def fmt(v: float) -> str:
    return "  n/a" if v != v else f"{v:5.3f}"


def report(name: str, m: Dict[str, float]) -> None:
    print(f"\n  {name}")
    print(f"    TP {m['tp']:5d}   FN {m['fn']:5d}  <- MISSED WATER (the safety error)")
    print(f"    FP {m['fp']:5d}   TN {m['tn']:5d}")
    print(f"    precision {fmt(m['precision'])}   recall {fmt(m['recall'])}   "
          f"F1 {fmt(m['f1'])}   acc {fmt(m['accuracy'])}")


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True,
                    help="Dataset root containing images and masks")
    ap.add_argument("--images-sub", default=None, help="Image subdirectory name")
    ap.add_argument("--masks-sub", default=None, help="Mask subdirectory name")
    ap.add_argument("--subset", default=None,
                    help="Label for this run (e.g. ONR, OFR, foggy, night)")
    ap.add_argument("--limit", type=int, default=0, help="Cap images processed")
    ap.add_argument("--semantic", action="store_true", help="Enable CLIPSeg cue 7")
    ap.add_argument("--residual", action="store_true", help="Enable Intrinsic cue 8")
    ap.add_argument("--no-corrections", action="store_true",
                    help="Disable the hand-tuned non-linear corrections")
    ap.add_argument("--fit", action="store_true",
                    help="Fit cue weights + threshold and write water_params.json")
    args = ap.parse_args()

    subset = args.subset or os.path.basename(os.path.normpath(args.data))

    if args.no_corrections:
        water_config.APPLY_CORRECTIONS = False

    print(f"Water ensemble evaluation — subset '{subset}'")
    print(f"  config: {water_config.describe()}")

    prior_fn = None
    if args.semantic:
        import semantic_water
        if not semantic_water.HAS_CLIPSEG:
            print("  !! CLIPSeg unavailable — cue 7 disabled")
        else:
            prior_fn = semantic_water.water_prior_map
            print(f"  cue 7: {semantic_water.describe()}")

    resid_fn = None
    if args.residual:
        import intrinsic_cues
        if not intrinsic_cues.HAS_INTRINSIC:
            print("  !! Intrinsic unavailable — cue 8 disabled")
        else:
            resid_fn = intrinsic_cues.residual_energy_map
            print(f"  cue 8: {intrinsic_cues.describe()}")

    pairs = discover_pairs(args.data, args.images_sub, args.masks_sub)
    if not pairs:
        print(f"\nNo (image, mask) pairs found under {args.data}")
        print("Pass --images-sub / --masks-sub if the layout is unusual.")
        return
    if args.limit:
        pairs = pairs[:args.limit]

    rows: List[Dict[str, object]] = []
    n_skipped = 0

    for k, (img_path, mask_path) in enumerate(pairs, start=1):
        bgr = cv2.imread(img_path)
        if bgr is None:
            n_skipped += 1
            continue
        gt = load_gt_mask(mask_path, bgr.shape[0], bgr.shape[1])
        if gt is None:
            n_skipped += 1
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pos_regions = regions_from_mask(gt)
        if not pos_regions:
            n_skipped += 1
            continue

        # Once per image, never per region.
        prior = prior_fn(rgb) if prior_fn else None
        resid = resid_fn(rgb) if resid_fn else None

        for region in pos_regions:
            for label, m in ((1, region), (0, sample_negative(region, gt))):
                if m is None:
                    continue
                r = detect_water(rgb, m, water_prior_map=prior, residual_map=resid)
                rows.append({
                    "image": os.path.basename(img_path),
                    "subset": subset,
                    "label": label,
                    "prob": r["water_probability"],
                    "pred": int(r["is_water"]),
                    "area": int(m.sum()),
                    "edge_density": r["edge_density_score"],
                    "gradient": r["gradient_score"],
                    "specular": r["specular_score"],
                    "color": r["color_score"],
                    "saturation": r["saturation_score"],
                    "semantic": r["semantic_score"],
                    "residual": r["residual_score"],
                })

        if k % 25 == 0 or k == len(pairs):
            print(f"    {k}/{len(pairs)} images, {len(rows)} regions", flush=True)

    if not rows:
        print("\nNo regions scored.")
        return

    y_true = np.array([r["label"] for r in rows])
    y_pred = np.array([r["pred"] for r in rows])
    probs = np.array([r["prob"] for r in rows], dtype=float)

    print(f"\n{len(rows)} regions from {len(pairs) - n_skipped} images "
          f"({int(y_true.sum())} water / {int((1 - y_true).sum())} dry)")
    print(f"corrections: {'ON' if water_config.APPLY_CORRECTIONS else 'OFF'}")

    report(f"as-shipped (threshold {water_config.DECISION_THRESHOLD})",
           metrics(y_true, y_pred))

    # Threshold sweep: is the hand-set 0.35 anywhere near optimal?
    best_t, best_f1 = None, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        m = metrics(y_true, (probs > t).astype(int))
        if m["f1"] == m["f1"] and m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    if best_t is not None:
        report(f"best threshold on this set ({best_t:.2f})",
               metrics(y_true, (probs > best_t).astype(int)))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"regions_{subset}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {os.path.relpath(csv_path, PROJECT_DIR)}")

    if args.fit:
        fit_weights(rows, subset, best_t)


def fit_weights(rows: List[Dict[str, object]], subset: str,
                best_t: Optional[float]) -> None:
    """
    Refit cue weights by logistic regression on the per-cue scores.

    Reported honestly: this is a fit on the evaluation set unless the caller
    supplies a held-out split, so the numbers above are optimistic. With a set
    the size of Puddle-1000 the right follow-up is a proper train/test split;
    this establishes that fitted weights beat judgement-set ones at all.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    cue_names = [c for c in ("edge_density", "gradient", "specular", "color",
                             "saturation", "semantic", "residual")
                 if rows[0].get(c) is not None]

    X = np.array([[float(r[c]) for c in cue_names] for r in rows])
    y = np.array([r["label"] for r in rows])

    if len(np.unique(y)) < 2:
        print("\n  cannot fit: only one class present")
        return

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    cv = cross_val_score(clf, X, y, cv=5, scoring="f1")
    clf.fit(X, y)

    coef = clf.coef_[0]
    print(f"\n  fitted on {len(cue_names)} cues, 5-fold CV F1 "
          f"{cv.mean():.3f} +/- {cv.std():.3f}")
    print(f"  {'cue':14s}{'hand-set':>10s}{'fitted':>10s}{'coef':>9s}")
    pos = np.clip(coef, 0, None)
    norm = pos / pos.sum() if pos.sum() > 0 else pos
    fitted = {}
    for name, c, nw in zip(cue_names, coef, norm):
        fitted[name] = round(float(nw), 4)
        print(f"  {name:14s}{water_config.CUE_WEIGHTS.get(name, 0):10.3f}"
              f"{nw:10.3f}{c:9.3f}")

    neg = [n for n, c in zip(cue_names, coef) if c < 0]
    if neg:
        print(f"\n  NOTE: negative coefficients on {', '.join(neg)} — these cues "
              "argue AGAINST water\n  on this set. The signed logistic form below "
              "preserves that; the legacy\n  non-negative `cue_weights` cannot and "
              "clips them to zero.")

    payload = {
        # Signed form — what detect_water actually uses when present. Negative
        # coefficients are preserved rather than clipped, which is the whole
        # point: `saturation` genuinely argues against water on real data.
        "logistic": {
            "coef": {n: round(float(c), 6) for n, c in zip(cue_names, coef)},
            "intercept": round(float(clf.intercept_[0]), 6),
        },
        # Legacy non-negative weights, retained so older readers and the
        # reporting tables still work. NOT used when `logistic` is present.
        "cue_weights": fitted,
        # 0.5, not `best_t`: the logistic output is a calibrated probability, so
        # the natural operating point is 0.5. `best_t` was tuned against the
        # weighted-mean output and does not transfer.
        "decision_threshold": 0.5,
        "apply_corrections": False,
        "n_labelled": len(rows),
        "dataset": subset,
        "fitted_at": date.today().isoformat(),
        "cv_f1_mean": round(float(cv.mean()), 4),
        "cv_f1_std": round(float(cv.std()), 4),
        "note": ("Fitted by scripts/eval_water_puddle1000.py --fit. Corrections "
                 "disabled because the fit subsumes them. `logistic` is the "
                 "authoritative form and admits negative coefficients; "
                 "`cue_weights` is the clipped non-negative legacy view. "
                 "Threshold is 0.5 because the logistic output is a calibrated "
                 "probability, not a weighted mean."),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "water_params.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {os.path.relpath(out, PROJECT_DIR)}")
    print("  water_config picks this up automatically on next import.")


if __name__ == "__main__":
    main()
