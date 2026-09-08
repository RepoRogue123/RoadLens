"""
Ablate the Phase 3 semantic override — does it actually help?

Compares three configurations on the hand-labelled set:

  off          no semantic override at all (the pre-Phase-3 baseline)
  one-way      the original rule: raw whole-scene ratio < 0.9, downgrade only
  calibrated   the corrected ring-local ratio with the fitted threshold,
               two-way, with the sparse-patch abstention

Measured against the illusion labels rather than against global severity ground
truth, which does not exist. This sidesteps the circularity that affects the
severity classifiers: we are asking "does the rule identify illusions", a
question the labels can actually answer.

The headline number is not accuracy. It is FALSE DOWNGRADES — real craters
wrongly called Shallow — because that is the error that hides a hazard.

Usage:
    python scripts/ablate_semantic.py
"""
import csv
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import semantic_config  # noqa: E402

CACHE_CSV = os.path.join(PROJECT_DIR, "data", "illusion_features_cache.csv")
LABELS_CSV = os.path.join(PROJECT_DIR, "data", "illusion_labels.csv")
OUT_CSV = os.path.join(PROJECT_DIR, "ml_results", "phase3_calibration", "ablation.csv")

LEGACY_RATIO = semantic_config.DEFAULT_DOWNGRADE_RATIO  # 0.9, whole-scene


def evaluate(rows, ratio_field, threshold, min_patches, name):
    """
    Returns counts for the illusion-detection decision.
      tp illusion correctly downgraded     fp real crater wrongly downgraded
      fn illusion missed                   tn real crater correctly untouched
      abstain sparse-patch declines (counted separately, never as a decision)
    """
    tp = fp = fn = tn = abstain = 0
    for r in rows:
        is_ill = r["label"] == "texture_illusion"
        val = r.get(ratio_field)
        patches = r.get("patches", 0)

        if val is None or val != val:
            abstain += 1
            continue
        if min_patches and patches < min_patches:
            abstain += 1
            continue

        fired = val < threshold
        if fired and is_ill:
            tp += 1
        elif fired and not is_ill:
            fp += 1
        elif not fired and is_ill:
            fn += 1
        else:
            tn += 1

    decided = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return dict(config=name, tp=tp, fp=fp, fn=fn, tn=tn, abstain=abstain,
                decided=decided, precision=prec, recall=rec, f1=f1)


def fmt(v):
    return "  n/a" if v != v else f"{v:5.3f}"


def main():
    if not os.path.isfile(LABELS_CSV):
        print("No labels. Run:  python scripts/label_illusions.py")
        return
    if not os.path.isfile(CACHE_CSV):
        print("No feature cache. Run:  python scripts/calibrate_semantic.py")
        return

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        labels = {r["image"]: r for r in csv.DictReader(f)
                  if r["label"] in ("real_crater", "texture_illusion")}
    with open(CACHE_CSV, newline="", encoding="utf-8") as f:
        feats = {r["image"]: r for r in csv.DictReader(f)}

    rows = []
    for name, lab in labels.items():
        fr = feats.get(name)
        if not fr:
            continue
        def num(k):
            try:
                return float(fr.get(k, "nan"))
            except ValueError:
                return float("nan")
        rows.append({
            "label": lab["label"],
            "stratum": lab["stratum"],
            "ratio_corrected": num("ratio"),
            "ratio_raw": num("ratio_raw"),
            "patches": int(float(fr.get("patches") or 0)),
        })

    if not rows:
        print("No rows with both a label and cached features.")
        return

    n_ill = sum(1 for r in rows if r["label"] == "texture_illusion")
    print(f"Ablation over {len(rows)} labelled potholes "
          f"({len(rows)-n_ill} real_crater / {n_ill} texture_illusion)")
    print(f"Calibration state: {semantic_config.describe()}\n")

    results = [
        # 'off' never fires: threshold of -inf means no downgrade ever happens.
        evaluate(rows, "ratio_corrected", float("-inf"), 0, "off (no override)"),
        evaluate(rows, "ratio_raw", LEGACY_RATIO, 0, f"one-way raw < {LEGACY_RATIO}"),
        evaluate(rows, "ratio_corrected", semantic_config.DOWNGRADE_RATIO,
                 semantic_config.MIN_PATCHES,
                 f"calibrated ring < {semantic_config.DOWNGRADE_RATIO:.3f}"),
    ]

    hdr = f"{'configuration':<32}{'caught':>7}{'missed':>7}{'FALSE':>7}{'ok':>5}{'absta':>7}{'prec':>7}{'rec':>7}"
    print(hdr)
    print(f"{'':32}{'(TP)':>7}{'(FN)':>7}{'DOWN':>7}{'(TN)':>5}{'in':>7}")
    print("-" * len(hdr))
    for r in results:
        print(f"{r['config']:<32}{r['tp']:>7}{r['fn']:>7}{r['fp']:>7}{r['tn']:>5}"
              f"{r['abstain']:>7}{fmt(r['precision']):>7}{fmt(r['recall']):>7}")

    print("\nFALSE DOWN = real craters wrongly downgraded to Shallow. This is the")
    print("safety-relevant error and should be the primary basis for judging the rule.")

    base_fp = results[1]["fp"]
    cal_fp = results[2]["fp"]
    if cal_fp < base_fp:
        print(f"\n  Calibration reduced false downgrades {base_fp} -> {cal_fp}.")
    elif cal_fp == base_fp:
        print(f"\n  False downgrades unchanged at {cal_fp}.")
    else:
        print(f"\n  ⚠  Calibration INCREASED false downgrades {base_fp} -> {cal_fp}.")
        print("     Consider raising --fp-cost in calibrate_semantic.py and refitting.")

    if results[2]["tp"] == 0 and results[2]["fp"] == 0:
        print("\n  ⚠  The calibrated rule never fired on this set. Either the threshold")
        print("     is too conservative or the signal is absent — check the ROC AUC.")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n  wrote {os.path.relpath(OUT_CSV, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
