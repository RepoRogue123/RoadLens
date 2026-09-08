"""
Fit the Phase 3 semantic-override threshold from labelled data.

Replaces the hand-set 0.9 variance ratio with an operating point chosen on
evidence. Reads data/illusion_labels.csv (produced by label_illusions.py),
recomputes the DINOv2 statistics for each labelled pothole, and sweeps the
threshold to produce ROC / PR curves.

COST ASYMMETRY — why we do not optimise accuracy
------------------------------------------------
The two errors are not equally bad:

  false downgrade   a real deep pothole is called Shallow because the rule
                    misfired. This is a SAFETY error — the system hides a
                    genuine hazard.
  missed illusion   a flat patch keeps an inflated verdict. This is a NUISANCE
                    error — someone inspects a road that turned out fine.

So the objective weights false downgrades more heavily (default 3:1). The
chosen point is reported alongside the full curve so the trade-off is visible
rather than buried in a single number.

The random and enriched strata are reported separately: the enriched stratum was
deliberately over-sampled for illusions, so its class balance is not the real
prior and precision computed on it would flatter the result.

Usage:
    python scripts/calibrate_semantic.py
    python scripts/calibrate_semantic.py --fp-cost 5      # even more cautious
"""
import argparse
import csv
import datetime
import glob
import json
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from segmentation import get_all_masks                        # noqa: E402
from foundation_features import extract_foundation_features   # noqa: E402
import semantic_config                                        # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "illusion_labels.csv")
CACHE_CSV = os.path.join(PROJECT_DIR, "data", "illusion_features_cache.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "phase3_calibration")

DARK, PANEL, TEXT = "#0f172a", "#1e293b", "#e2e8f0"
RATIO_KEY = "dinov2_ratio_corrected"
MIN_ILLUSIONS = 15


def find_image(name):
    for d in ("data1/train/images", "merged_dataset/train/images",
              "data1/valid/images", "merged_dataset/valid/images"):
        p = os.path.join(PROJECT_DIR, d, name)
        if os.path.isfile(p):
            return p
    hits = glob.glob(os.path.join(PROJECT_DIR, "**", name), recursive=True)
    return hits[0] if hits else None


def compute_features(rows):
    """Attach DINOv2 statistics to each labelled row, caching across runs."""
    cache = {}
    if os.path.isfile(CACHE_CSV):
        with open(CACHE_CSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cache[r["image"]] = r

    out, computed = [], 0
    for i, row in enumerate(rows, 1):
        name = row["image"]
        if name in cache:
            c = cache[name]
            try:
                row["ratio"] = float(c["ratio"])
                row["patches"] = int(c["patches"])
                row["ratio_raw"] = float(c.get("ratio_raw", "nan"))
                out.append(row)
            except (ValueError, KeyError):
                pass
            continue

        path = find_image(name)
        if not path:
            print(f"  ! image not found, skipping: {name}")
            continue
        try:
            masks = get_all_masks(path)
            if not masks:
                continue
            img = cv2.imread(path)
            feats = extract_foundation_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), masks[0])
            if not feats:
                continue
            iv = feats["dinov2_inside_variance"]
            ov = feats["dinov2_outside_variance"]
            row["ratio"] = feats.get(RATIO_KEY, float("nan"))
            row["ratio_raw"] = iv / ov if ov > 1e-9 else float("nan")
            row["patches"] = feats.get("dinov2_patch_count", 0)
            out.append(row)
            computed += 1
            if computed % 10 == 0:
                print(f"  computed {computed} ({i}/{len(rows)})")
        except Exception as e:
            print(f"  ! failed on {name}: {e}")

    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "ratio", "ratio_raw", "patches"])
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    return out


def sweep(ratios, is_illusion, fp_cost):
    """
    Sweep the downgrade threshold. Predicting 'illusion' means ratio < t.
      TP illusion correctly caught      FP real crater wrongly downgraded (costly)
      FN illusion missed                TN real crater correctly left alone
    """
    grid = np.unique(np.concatenate([np.linspace(0.2, 2.0, 361), ratios]))
    rows = []
    for t in grid:
        pred = ratios < t
        tp = int(np.sum(pred & is_illusion))
        fp = int(np.sum(pred & ~is_illusion))
        fn = int(np.sum(~pred & is_illusion))
        tn = int(np.sum(~pred & ~is_illusion))
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        tpr = rec
        fpr = fp / (fp + tn) if fp + tn else 0.0
        cost = fp_cost * fp + fn
        rows.append(dict(t=float(t), tp=tp, fp=fp, fn=fn, tn=tn,
                         precision=prec, recall=rec, tpr=tpr, fpr=fpr, cost=cost))
    return rows


def auc(fpr, tpr):
    o = np.argsort(fpr)
    return float(np.trapezoid(np.array(tpr)[o], np.array(fpr)[o]))


def plot(rows, best, ratios, is_illusion, auc_val, n_lab):
    os.makedirs(OUT_DIR, exist_ok=True)
    fpr = [r["fpr"] for r in rows]; tpr = [r["tpr"] for r in rows]
    rec = [r["recall"] for r in rows]; prec = [r["precision"] for r in rows]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=DARK)
    for ax in (a1, a2, a3):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for s in ax.spines.values():
            s.set_color("#334155")

    a1.plot(fpr, tpr, color="#22d3ee", lw=2)
    a1.plot([0, 1], [0, 1], "--", color="#64748b", lw=1)
    a1.scatter([best["fpr"]], [best["tpr"]], color="#f59e0b", s=90, zorder=5)
    a1.set_title(f"ROC  ·  AUC = {auc_val:.3f}", color=TEXT, fontsize=12)
    a1.set_xlabel("false positive rate", color="#94a3b8")
    a1.set_ylabel("true positive rate", color="#94a3b8")

    a2.plot(rec, prec, color="#a78bfa", lw=2)
    a2.scatter([best["recall"]], [best["precision"]], color="#f59e0b", s=90, zorder=5)
    a2.set_title("Precision / Recall", color=TEXT, fontsize=12)
    a2.set_xlabel("recall (illusions caught)", color="#94a3b8")
    a2.set_ylabel("precision", color="#94a3b8")

    bins = np.linspace(min(ratios.min(), 0.2), max(ratios.max(), 1.6), 26)
    a3.hist(ratios[~is_illusion], bins=bins, color="#34d399", alpha=0.75, label="real crater")
    a3.hist(ratios[is_illusion], bins=bins, color="#f87171", alpha=0.75, label="texture illusion")
    a3.axvline(best["t"], color="#f59e0b", ls="--", lw=2)
    a3.axvline(semantic_config.DEFAULT_DOWNGRADE_RATIO, color="#94a3b8", ls=":", lw=1.5)
    a3.set_title(f"Separation  (fitted {best['t']:.3f}, old 0.9 dotted)", color=TEXT, fontsize=12)
    a3.set_xlabel("variance ratio (interior / ring)", color="#94a3b8")
    a3.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT, fontsize=9)

    fig.suptitle(
        f"PHASE 3 CALIBRATION  ·  n={n_lab} labelled  ·  "
        f"chosen point: precision {best['precision']:.2f}, recall {best['recall']:.2f}, "
        f"{best['fp']} false downgrade(s)",
        color=TEXT, fontsize=13, y=0.99,
    )
    p = os.path.join(OUT_DIR, "phase3_calibration.png")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(p, dpi=150, facecolor=DARK, bbox_inches="tight")
    plt.close(fig)
    print("  saved", os.path.relpath(p, PROJECT_DIR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-cost", type=float, default=3.0,
                    help="cost of a false downgrade relative to a missed illusion")
    ap.add_argument("--stratum", choices=["all", "random", "enriched"], default="all")
    args = ap.parse_args()

    if not os.path.isfile(LABELS_CSV):
        print(f"No labels at {os.path.relpath(LABELS_CSV, PROJECT_DIR)}.")
        print("Run:  python scripts/label_illusions.py")
        return

    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in ("real_crater", "texture_illusion")]
    if args.stratum != "all":
        rows = [r for r in rows if r["stratum"] == args.stratum]

    n_ill = sum(1 for r in rows if r["label"] == "texture_illusion")
    n_real = len(rows) - n_ill
    print(f"{len(rows)} usable labels  ({n_real} real_crater / {n_ill} texture_illusion)")
    if n_ill < MIN_ILLUSIONS or n_real < MIN_ILLUSIONS:
        print(f"\n  ⚠  Need at least {MIN_ILLUSIONS} of each class for a meaningful fit.")
        print("     Label more with:  python scripts/label_illusions.py")
        if n_ill == 0 or n_real == 0:
            return
        print("     Continuing anyway — treat the result as indicative only.\n")

    print("\nComputing DINOv2 statistics…")
    rows = compute_features(rows)
    rows = [r for r in rows if r.get("ratio") == r.get("ratio")]  # drop NaN
    if not rows:
        print("No usable feature rows.")
        return

    ratios = np.array([r["ratio"] for r in rows], dtype=float)
    is_ill = np.array([r["label"] == "texture_illusion" for r in rows])

    swept = sweep(ratios, is_ill, args.fp_cost)
    auc_val = auc([r["fpr"] for r in swept], [r["tpr"] for r in swept])
    best = min(swept, key=lambda r: (r["cost"], -r["recall"]))

    # Upgrade threshold: the ratio above which real craters dominate.
    real_ratios = ratios[~is_ill]
    upgrade = float(np.percentile(real_ratios, 60)) if len(real_ratios) >= 5 \
        else semantic_config.DEFAULT_UPGRADE_RATIO

    print(f"\nAUC = {auc_val:.3f}")
    if auc_val < 0.6:
        print("  ⚠  AUC near chance — the DINOv2 variance ratio is NOT separating these")
        print("     classes. That is a legitimate negative result and should be reported")
        print("     as one rather than papered over with a tuned threshold.")

    print(f"\nChosen operating point (false-downgrade cost {args.fp_cost}:1)")
    print(f"  downgrade when ratio < {best['t']:.3f}   (was {semantic_config.DEFAULT_DOWNGRADE_RATIO})")
    print(f"  precision {best['precision']:.3f}   recall {best['recall']:.3f}")
    print(f"  illusions caught {best['tp']}/{best['tp']+best['fn']}   "
          f"real craters wrongly downgraded {best['fp']}")
    print(f"  upgrade when ratio > {upgrade:.3f}")

    plot(swept, best, ratios, is_ill, auc_val, len(rows))

    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "downgrade_ratio": round(best["t"], 4),
        "upgrade_ratio": round(upgrade, 4),
        "min_patches": semantic_config.DEFAULT_MIN_PATCHES,
        "ratio_key": RATIO_KEY,
        "n_labelled": len(rows),
        "n_illusion": int(is_ill.sum()),
        "n_real": int((~is_ill).sum()),
        "auc": round(auc_val, 4),
        "precision": round(best["precision"], 4),
        "recall": round(best["recall"], 4),
        "false_downgrades": best["fp"],
        "fp_cost": args.fp_cost,
        "stratum": args.stratum,
        "fitted_at": datetime.date.today().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "threshold.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(OUT_DIR, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(swept[0].keys()))
        w.writeheader()
        w.writerows(swept)

    print(f"\n  wrote {os.path.relpath(OUT_DIR, PROJECT_DIR)}/threshold.json")
    print("  semantic_config.py will now pick this up automatically.")


if __name__ == "__main__":
    main()
