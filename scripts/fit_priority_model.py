"""
Method 2B: learn the risk matrix's CONSEQUENCE (1-5) from the team's ratings.

Labels: data/consequence_labels.csv (scripts/merge_annotations.py on a consequence pack);
measurements: data/hazard_pool.csv (scripts/build_hazard_pool.py), joined by pool_id.

Model: ordinal logistic regression in its simplest form, one logistic model per threshold,
P(consequence > k) for k = 1..4, on the seven standardised measurements of hazard.FEATURES
(missing values imputed with the training median). The predicted consequence is the
expected value 1 + sum_k P(> k), rounded. Weights are fitted, never set by hand.

Evaluation: cross-fitted over 5 folds grouped by PHOTO (two potholes from one photo never
split across train and test), scored with quadratically weighted kappa against the team's
ratings (rounded half up). The same score is computed for the provisional rule it would
replace (hazard.consequence_from_measurements, with no wheel drop since pool photos have no
camera distance).

ADOPTION RULE (fixed before any labels exist): at least 100 rated potholes, and the 95%
bootstrap interval (2,000 resamples of potholes) of kappa(model) - kappa(provisional rule)
lies above zero. Only then is ml_models/hazard/priority_model.json written with
"adopted": true; hazard.learned_consequence ignores any other file.

Usage:
    python scripts/fit_priority_model.py [--labels data/consequence_labels.csv] [--dry-run]
"""
import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import hazard                                                                    # noqa: E402
from scripts.build_hazard_pool import POOL_CSV                                   # noqa: E402
from scripts.cvpr_severity_framework import depth_samples, GRADING_SEED         # noqa: E402
from scripts.merge_annotations import weighted_kappa                             # noqa: E402

LABELS_CSV = os.path.join(PROJECT_DIR, "data", "consequence_labels.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "hazard")
MIN_LABELS = 100
N_BOOT = 2000
FOLDS = 5


# ── shared with scripts/fit_hazard_pairs.py ───────────────────────────────────
def scaler(X):
    """Median imputation and standardisation fitted on X: (impute_median, mean, scale)."""
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xi = np.where(np.isfinite(X), X, med)
    sd = Xi.std(0)
    return med, Xi.mean(0), np.where(sd > 0, sd, 1.0)


def transform(X, med, mean, scale):
    return (np.where(np.isfinite(X), X, med) - mean) / scale


def load_pool():
    if not os.path.isfile(POOL_CSV):
        sys.exit("data/hazard_pool.csv missing - run scripts/build_hazard_pool.py")
    return pd.read_csv(POOL_CSV).set_index("pool_id")


def bootstrap_diff(a, b, rng, stat):
    """95% interval of stat(a) - stat(b), resampling the same items for both."""
    n = len(a[0])
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        d.append(stat(*[x[i] for x in a]) - stat(*[x[i] for x in b]))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


# ── ordinal model ─────────────────────────────────────────────────────────────
def fit_ordinal(Z, y):
    """Per threshold k=1..4: (weights, bias) of P(y > k). A threshold no label crosses is a constant."""
    W, B = [], []
    for k in range(1, 5):
        t = (y > k).astype(int)
        if t.min() == t.max():
            p = np.clip(t.mean(), 1e-3, 1 - 1e-3)
            W.append(np.zeros(Z.shape[1]))
            B.append(float(np.log(p / (1 - p))))
            continue
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Z, t)
        W.append(m.coef_[0])
        B.append(float(m.intercept_[0]))
    return np.array(W), np.array(B)


def predict_ordinal(W, B, Z):
    above = 1.0 / (1.0 + np.exp(-(Z @ W.T + B)))
    above = np.minimum.accumulate(above, axis=1)          # P(> k) cannot rise with k
    return 1.0 + above.sum(1)


def to_grade(v):
    return np.clip(np.floor(np.asarray(v, dtype=np.float64) + 0.5), 1, 5).astype(int)


def provisional(pool_rows, rng):
    """hazard.consequence_from_measurements for each pothole (no camera distance: no wheel drop)."""
    out = []
    for _, r in pool_rows.iterrows():
        true, _ = depth_samples(float(r.depth_mm), bool(r.familiar), rng)
        p = float((true >= hazard.INVESTIGATORY["carriageway"]["depth_mm"]).mean())
        out.append(hazard.consequence_from_measurements(p, None)["score"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=LABELS_CSV)
    ap.add_argument("--dry-run", action="store_true", help="report only; write no model")
    args = ap.parse_args()
    if not os.path.isfile(args.labels):
        sys.exit(f"{args.labels} missing - merge a consequence pack first (scripts/merge_annotations.py)")
    pool = load_pool()
    lab = pd.read_csv(args.labels)
    lab = lab[lab.pool_id.isin(pool.index)]
    rows = pool.loc[lab.pool_id]
    X = rows[hazard.FEATURES].to_numpy(np.float64)
    y = lab.label.astype(float).to_numpy()
    groups = rows.image.to_numpy()
    n = len(y)
    print(f"{n} rated potholes from {len(set(groups))} photos; ratings {({int(k): int(v) for k, v in pd.Series(to_grade(y)).value_counts().sort_index().items()})}")
    if n < 2 * FOLDS:
        sys.exit("too few ratings to cross-fit")

    pred = np.zeros(n)
    for tr, te in GroupKFold(n_splits=FOLDS).split(X, y, groups):
        s = scaler(X[tr])
        W, B = fit_ordinal(transform(X[tr], *s), y[tr])
        pred[te] = predict_ordinal(W, B, transform(X[te], *s))
    rng = np.random.default_rng(GRADING_SEED)
    rule = provisional(rows, rng)
    truth = to_grade(y)
    k_model = weighted_kappa(truth, to_grade(pred))
    k_rule = weighted_kappa(truth, rule)
    ci = bootstrap_diff((truth, to_grade(pred)), (truth, rule), rng, weighted_kappa)
    adopted = n >= MIN_LABELS and ci[0] > 0
    print(f"cross-fitted weighted kappa: model {k_model:.2f}, provisional rule {k_rule:.2f}, "
          f"difference 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
    print(f"mean absolute error (1-5 scale): model {np.abs(pred - y).mean():.2f}, rule {np.abs(rule - y).mean():.2f}")

    s = scaler(X)
    W, B = fit_ordinal(transform(X, *s), y)
    by_source = {src: {"n": int(m.sum()),
                       "kappa_model": float(weighted_kappa(truth[m], to_grade(pred[m]))) if m.sum() > 1 else None}
                 for src in sorted(set(rows.source)) for m in [(rows.source == src).to_numpy()]}
    report = {"fitted": str(date.today()), "n_labels": n, "cv_kappa": k_model, "rule_kappa": k_rule,
              "kappa_diff_ci95": ci, "adopted": bool(adopted), "by_source": by_source,
              "adoption_rule": f">= {MIN_LABELS} labels and the 95% CI of kappa(model) - kappa(rule) above 0",
              "weights_full_fit": {f: [round(float(w), 4) for w in W[:, j]] for j, f in enumerate(hazard.FEATURES)}}
    print("ADOPTED" if adopted else f"not adopted ({report['adoption_rule']})")
    if args.dry_run:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "priority_model_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    model = {"features": hazard.FEATURES, "impute_median": s[0].tolist(), "mean": s[1].tolist(),
             "scale": s[2].tolist(), "weights": W.tolist(), "bias": B.tolist(), "n_labels": n,
             "cv_kappa": k_model, "adopted": bool(adopted), "fitted": str(date.today())}
    os.makedirs(hazard.MODEL_DIR, exist_ok=True)
    with open(hazard.PRIORITY_MODEL, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    print(f"wrote {os.path.relpath(hazard.PRIORITY_MODEL, PROJECT_DIR)} and ml_results/hazard/priority_model_report.json")


if __name__ == "__main__":
    main()
