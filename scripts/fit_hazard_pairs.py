"""
Method 3B: learn a hazard ranking from the team's "which pothole is more dangerous?" answers.

Labels: data/hazard_pair_labels.csv (scripts/merge_annotations.py on a hazard_pairs pack);
measurements: data/hazard_pool.csv, joined by pool_id.

Model: Bradley-Terry with a linear score, s = w . z over the seven standardised
measurements of hazard.FEATURES, so P(A more dangerous than B) = sigmoid(s_A - s_B). This
is a logistic regression without intercept on the difference of the two potholes'
measurements. "equal" answers count as half a win each way. Weights are fitted, never set.

Evaluation: cross-fitted over 5 folds of POTHOLES: a pair is scored by a model trained on
pairs that contain neither of its potholes. Score: agreement, the share of the team's
decided pairs (A or B) where the model ranks the same pothole higher. Baselines, scored the
same way: each single measurement on its own, including the deepest point, which is the
quantity that bounds every wheel drop (Method 3A) when no scale is known.

ADOPTION RULE (fixed before any labels exist): at least 150 decided pairs, and the 95%
bootstrap interval (2,000 resamples of pairs) of agreement(model) - agreement(best single
measurement) lies above zero. Only then is ml_models/hazard/hazard_pairs_model.json written
with "adopted": true; hazard.learned_hazard ignores any other file.

Usage:
    python scripts/fit_hazard_pairs.py [--labels data/hazard_pair_labels.csv] [--dry-run]
"""
import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import hazard                                                                    # noqa: E402
from scripts.fit_priority_model import OUT_DIR, bootstrap_diff, load_pool, scaler, transform  # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "hazard_pair_labels.csv")
MIN_PAIRS = 150
FOLDS = 5
SEED = 20261013


def fit_bt(Za, Zb, y, w):
    """Weights of the Bradley-Terry score from paired standardised measurements."""
    d = np.vstack([Za - Zb, Zb - Za])                         # both orders: no intercept needed
    t = np.concatenate([y, 1 - y]).round().astype(int)
    sw = np.concatenate([w, w])
    m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(d, t, sample_weight=sw)
    return m.coef_[0]


def expand(lab):
    """(pool_a, pool_b, y, weight): A win y=1, B win y=0, equal -> one half-weight row each way."""
    rows = []
    for r in lab.itertuples():
        if r.label == "a":
            rows.append((r.pool_a, r.pool_b, 1.0, 1.0))
        elif r.label == "b":
            rows.append((r.pool_a, r.pool_b, 0.0, 1.0))
        elif r.label == "equal":
            rows += [(r.pool_a, r.pool_b, 1.0, 0.5), (r.pool_a, r.pool_b, 0.0, 0.5)]
    return pd.DataFrame(rows, columns=["a", "b", "y", "w"])


def agreement(margin, y):
    """Share of decided pairs ranked the same way; ties count half."""
    return float(np.mean(np.where(margin == 0, 0.5, (margin > 0) == (y == 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not os.path.isfile(args.labels):
        sys.exit(f"{args.labels} missing - merge a hazard_pairs pack first (scripts/merge_annotations.py)")
    pool = load_pool()
    lab = pd.read_csv(args.labels)
    lab = lab[lab.pool_a.isin(pool.index) & lab.pool_b.isin(pool.index)]
    pairs = expand(lab)
    X = pool[hazard.FEATURES].to_numpy(np.float64)
    pos = {pid: i for i, pid in enumerate(pool.index)}
    ia, ib = pairs.a.map(pos).to_numpy(), pairs.b.map(pos).to_numpy()
    decided = (pairs.w == 1.0).to_numpy()
    print(f"{len(lab)} answered pairs: {int(decided.sum())} decided, {int((lab.label == 'equal').sum())} equal")
    if decided.sum() < 2 * FOLDS:
        sys.exit("too few decided pairs to cross-fit")

    rng = np.random.default_rng(SEED)
    fold = rng.integers(0, FOLDS, len(pool))
    margin_sum, margin_n = np.zeros(len(pairs)), np.zeros(len(pairs))
    for k in range(FOLDS):
        test_p = fold == k
        tr = ~(test_p[ia] | test_p[ib])
        te = test_p[ia] | test_p[ib]
        s = scaler(X[np.unique(np.concatenate([ia[tr], ib[tr]]))])
        Z = transform(X, *s)
        w = fit_bt(Z[ia[tr]], Z[ib[tr]], pairs.y.to_numpy()[tr], pairs.w.to_numpy()[tr])
        margin_sum[te] += (Z[ia[te]] - Z[ib[te]]) @ w
        margin_n[te] += 1
    margin = margin_sum / np.maximum(margin_n, 1)
    y = pairs.y.to_numpy()
    agree = agreement(margin[decided], y[decided])

    base = {}
    for j, f in enumerate(hazard.FEATURES):
        xa, xb = X[ia, j], X[ib, j]
        ok = decided & np.isfinite(xa) & np.isfinite(xb)
        base[f] = {"agreement": agreement((xa - xb)[ok], y[ok]), "n": int(ok.sum())}
    best = max((f for f in base if base[f]["n"] == decided.sum()), key=lambda f: base[f]["agreement"],
               default="depth_mm")
    xa, xb = X[ia, hazard.FEATURES.index(best)], X[ib, hazard.FEATURES.index(best)]
    ci = bootstrap_diff((margin[decided], y[decided]), ((xa - xb)[decided], y[decided]), rng, agreement)
    adopted = decided.sum() >= MIN_PAIRS and ci[0] > 0
    print(f"held-out agreement with the team: model {agree:.1%}")
    for f, b in sorted(base.items(), key=lambda kv: -kv[1]["agreement"]):
        tag = "  <- 3A physics bound" if f == "deepest_mm" else ""
        print(f"  {f:15s} alone {b['agreement']:.1%} (n {b['n']}){tag}")
    print(f"model - best single ({best}) 95% CI [{ci[0]:+.1%}, {ci[1]:+.1%}]")
    print("ADOPTED" if adopted else f"not adopted (needs >= {MIN_PAIRS} decided pairs and CI above 0)")

    s = scaler(X)
    Z = transform(X, *s)
    w = fit_bt(Z[ia], Z[ib], y, pairs.w.to_numpy())
    report = {"fitted": str(date.today()), "n_pairs": int(decided.sum()), "cv_agreement": agree,
              "single_measurement_agreement": base, "best_single": best, "diff_ci95": ci,
              "adopted": bool(adopted), "weights_full_fit": dict(zip(hazard.FEATURES, np.round(w, 4).tolist()))}
    if args.dry_run:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "hazard_pairs_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    model = {"features": hazard.FEATURES, "impute_median": s[0].tolist(), "mean": s[1].tolist(),
             "scale": s[2].tolist(), "weights": w.tolist(), "pool_scores": sorted((Z @ w).round(4).tolist()),
             "n_pairs": int(decided.sum()), "cv_agreement": agree, "adopted": bool(adopted),
             "fitted": str(date.today())}
    os.makedirs(hazard.MODEL_DIR, exist_ok=True)
    with open(hazard.PAIRS_MODEL, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    print(f"wrote {os.path.relpath(hazard.PAIRS_MODEL, PROJECT_DIR)} and ml_results/hazard/hazard_pairs_report.json")


if __name__ == "__main__":
    main()
