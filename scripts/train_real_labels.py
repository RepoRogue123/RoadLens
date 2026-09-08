"""
M1.2 + M4.2 — train severity on MEASURED labels, and report honestly.

Why a new script rather than extending ml_classifier.py
-------------------------------------------------------
ml_classifier.py's loading path is built around KMeans pseudo-labels derived
from Depth-Anything, splits at random, and reports accuracy with no baseline.
Every one of those choices is exactly what this run needs to avoid. Bolting a
`--labels real` flag onto it would leave the misleading machinery in place; a
purpose-built script makes the discipline explicit.

The three rules this script enforces
------------------------------------
1. SPLIT BY FRAME. Two potholes in one photograph share lighting, camera pose,
   road surface and operator. Splitting by pothole leaks all of that across the
   boundary. `GroupShuffleSplit` on the frame key, always.

2. BASELINES IN EVERY TABLE. Without them a number means nothing, which is how
   the project ended up quoting 99.7%:
       majority class      ~43% by construction
       predict the median  MAE 13.78 mm
       log(area) alone     MAE 12.03 mm  — one free feature
   A model that does not beat these is reported as not beating them.

3. THREE FEATURE VARIANTS, compared head to head, because the backbone question
   is live: Depth-Anything scores Pearson 0.019 against real depth while MoGe-3
   scores 0.545.
       da_only            the 39 features as the pipeline computes them today
       moge_only          the same 39, but under MoGe metric depth
       moge_plus_geom     MoGe plus log(area) and the mask-only geometry

Usage:
    python scripts/train_real_labels.py
    python scripts/train_real_labels.py --task regression
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from sklearn.model_selection import GroupShuffleSplit          # noqa: E402
from sklearn.preprocessing import StandardScaler               # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge     # noqa: E402
from sklearn.ensemble import (                                 # noqa: E402
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC                                    # noqa: E402
from sklearn.naive_bayes import GaussianNB                     # noqa: E402
from sklearn.neighbors import KNeighborsClassifier             # noqa: E402
from sklearn.neural_network import MLPClassifier               # noqa: E402
from sklearn.metrics import accuracy_score, f1_score           # noqa: E402

from inference import FEATURE_COLS_20, GEOMETRY_FEATURE_COLS   # noqa: E402

CSV = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd",
                   "trainset_real_labels.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
ALL_COLS = FEATURE_COLS_20 + GEOMETRY_FEATURE_COLS
SEED = 20260909

# Mask-only columns: identical under either backend, so they belong to the
# "geometry" variant rather than to a depth backbone.
MASK_ONLY = [
    "height", "width", "box_area", "pothole_area", "nonpothole_area",
    "aspect_ratio", "solidity", "compactness", "surface_area_px2",
    "surface_area_cm2",
] + GEOMETRY_FEATURE_COLS[:10]          # the ten curvature columns


#  Columns that genuinely depend on the depth map. Everything else in the 39 is
#  computed from the mask alone and is byte-identical under either backend.
DEPTH_DERIVED = [c for c in ALL_COLS if c not in MASK_ONLY]


def variants(df: pd.DataFrame):
    """
    Feature subsets, arranged so the backbone question is actually separable.

    The first three variants share ~20 mask-only columns, which is why `da_only`
    and `moge_only` scored within 0.02 of each other despite Depth-Anything
    having r=0.019 against real depth and MoGe r=0.545. The last three isolate
    the parts that differ, so "does the depth map contribute anything at all"
    can be answered rather than assumed.
    """
    da = [f"da_{c}" for c in ALL_COLS if f"da_{c}" in df.columns]
    moge = [f"moge_{c}" for c in ALL_COLS if f"moge_{c}" in df.columns]
    geom = [f"moge_{c}" for c in MASK_ONLY if f"moge_{c}" in df.columns]
    return {
        "da_only": da,
        "moge_only": moge,
        "moge_plus_geom": sorted(set(moge + geom + ["log_area"])),
        # ── isolating variants ──
        "shape_only_NO_DEPTH": sorted(set(geom + ["log_area"])),
        "da_depth_only": [f"da_{c}" for c in DEPTH_DERIVED
                          if f"da_{c}" in df.columns],
        "moge_depth_only": [f"moge_{c}" for c in DEPTH_DERIVED
                            if f"moge_{c}" in df.columns],
    }


def split(df: pd.DataFrame, test_size=0.25):
    """Frame-level split. Never split on row index."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr, te = next(gss.split(df, groups=df["key"]))
    return df.iloc[tr], df.iloc[te]


def classification(df):
    tr, te = split(df)
    ytr = tr["gt_severity"].values
    yte = te["gt_severity"].values

    print(f"  train {len(tr)} potholes / {tr['key'].nunique()} frames")
    print(f"  test  {len(te)} potholes / {te['key'].nunique()} frames")
    assert not (set(tr["key"]) & set(te["key"])), "frame leaked across split"

    maj = pd.Series(ytr).mode()[0]
    base_acc = accuracy_score(yte, [maj] * len(yte))
    print(f"\n  BASELINE  majority class '{maj}': accuracy {base_acc:.3f}")
    print("  (any model at or below this has learned nothing)\n")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000,
                                                 class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200,
                                               class_weight="balanced",
                                               random_state=SEED),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=SEED),
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                             random_state=SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=200, random_state=SEED,
                                          verbosity=0)
    except Exception:
        pass

    results = {}
    for vname, cols in variants(df).items():
        if not cols:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(np.nan_to_num(tr[cols].values.astype(float)))
        Xte = sc.transform(np.nan_to_num(te[cols].values.astype(float)))

        print(f"  === {vname}  ({len(cols)} features) ===")
        print(f"  {'model':<20}{'test acc':>10}{'macro F1':>10}"
              f"{'vs base':>10}{'train acc':>11}")
        print("  " + "-" * 61)
        for mname, m in models.items():
            ytr_m, yte_m = ytr, yte
            if mname == "XGBoost":
                classes = sorted(set(ytr))
                idx = {c: i for i, c in enumerate(classes)}
                ytr_m = np.array([idx[v] for v in ytr])
                yte_m = np.array([idx[v] for v in yte])
            try:
                m.fit(Xtr, ytr_m)
                pred = m.predict(Xte)
                acc = accuracy_score(yte_m, pred)
                f1 = f1_score(yte_m, pred, average="macro")
                tracc = accuracy_score(ytr_m, m.predict(Xtr))
            except Exception as e:
                print(f"  {mname:<20}  failed: {type(e).__name__}")
                continue
            results[f"{vname}/{mname}"] = {"acc": acc, "f1": f1, "train": tracc}
            flag = "" if acc > base_acc + 0.02 else "  <- no better than baseline"
            print(f"  {mname:<20}{acc:>10.3f}{f1:>10.3f}"
                  f"{acc-base_acc:>+10.3f}{tracc:>11.3f}{flag}")
        print()
    return results, base_acc


def regression(df):
    tr, te = split(df)
    ytr = tr["gt_mm"].values.astype(float)
    yte = te["gt_mm"].values.astype(float)

    print(f"  train {len(tr)} / test {len(te)} potholes "
          f"({tr['key'].nunique()}/{te['key'].nunique()} frames)")

    med = float(np.median(ytr))
    mae_med = float(np.mean(np.abs(yte - med)))
    la_tr = tr["log_area"].values.astype(float)
    la_te = te["log_area"].values.astype(float)
    A = np.column_stack([la_tr, np.ones_like(la_tr)])
    c, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    mae_la = float(np.mean(np.abs(np.column_stack(
        [la_te, np.ones_like(la_te)]) @ c - yte)))

    print(f"\n  BASELINES on the held-out frames")
    print(f"    predict median ({med:.1f}mm)   MAE {mae_med:6.2f} mm")
    print(f"    log(area) alone              MAE {mae_la:6.2f} mm")
    print("  (a feature set that cannot beat log(area) is not earning its keep)\n")

    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300,
                                              random_state=SEED),
        "GradientBoosting": GradientBoostingRegressor(random_state=SEED),
    }

    results = {}
    for vname, cols in variants(df).items():
        if not cols:
            continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(np.nan_to_num(tr[cols].values.astype(float)))
        Xte = sc.transform(np.nan_to_num(te[cols].values.astype(float)))

        print(f"  === {vname}  ({len(cols)} features) ===")
        print(f"  {'model':<20}{'test MAE':>10}{'vs median':>11}"
              f"{'vs logarea':>12}{'train MAE':>11}")
        print("  " + "-" * 64)
        for mname, m in models.items():
            try:
                m.fit(Xtr, ytr)
                p = m.predict(Xte)
                mae = float(np.mean(np.abs(p - yte)))
                trmae = float(np.mean(np.abs(m.predict(Xtr) - ytr)))
            except Exception as e:
                print(f"  {mname:<20}  failed: {type(e).__name__}")
                continue
            results[f"{vname}/{mname}"] = {"mae": mae, "train_mae": trmae}
            flag = "" if mae < mae_la else "  <- worse than log(area) alone"
            print(f"  {mname:<20}{mae:>10.2f}{mae-mae_med:>+11.2f}"
                  f"{mae-mae_la:>+12.2f}{trmae:>11.2f}{flag}")
        print()

    return results, {"median": mae_med, "log_area": mae_la}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--task", choices=["both", "classification", "regression"],
                    default="both")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Missing {args.csv}\nRun scripts/build_pothrgbd_trainset.py first.")
        return

    df = pd.read_csv(args.csv)
    print(f"Training on MEASURED labels — {len(df)} potholes, "
          f"{df['key'].nunique()} frames")
    print(f"  gt depth median {df['gt_mm'].median():.1f}mm")
    print(f"  {dict(df['gt_severity'].value_counts())}\n")

    out = {}
    if args.task in ("both", "classification"):
        print("=" * 68)
        print("CLASSIFICATION — 3-class severity from measured mm bands")
        print("=" * 68)
        r, b = classification(df)
        out["classification"] = {"results": r, "baseline_acc": b}

    if args.task in ("both", "regression"):
        print("=" * 68)
        print("REGRESSION — bowl depth in millimetres  (M4.2)")
        print("=" * 68)
        r, b = regression(df)
        out["regression"] = {"results": r, "baselines": b}

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "real_label_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {os.path.relpath(p, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
