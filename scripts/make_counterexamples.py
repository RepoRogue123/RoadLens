"""
Evidence that the old approach failed: one counterexample per old method, plus the numbers
behind the graphs, all on the SAME held-out measured potholes.

Population: the 208 measured PothRGBD potholes in the test capture sessions (human outlines),
which neither the segmenter nor the served depth networks trained on. The old served verdict
is the one recorded for 742 measured potholes (scripts/compare_served_severity.py).

Counterexamples are chosen by fixed rules, written below before looking at any picture, and
each comes with how often that failure happens across the population, so no example stands
alone:

    old verdict, too high   measured < 15 mm, old system said Deep          (smallest measured depth)
    old verdict, too low    measured >= 50 mm, DINOv2 check downgraded it   (deepest)
    Depth-Anything          a measured-deep pothole it ranks among the shallowest quarter,
                            beside a measured-shallow one it ranks among the deepest quarter
    curvature               a top-quarter curvature pothole that is shallow, beside a
                            bottom-quarter curvature pothole that is deep
    MoGe-3 raw              the largest raw over-estimate among potholes measured >= 10 mm
    shape-from-shading      the most negative correlation with measured relief, among
                            potholes measured >= 30 mm

Writes ml_results/figures/counterexamples/*.png and summary.json (chart data included).

Usage:
    python scripts/make_counterexamples.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from metric_features import road_plane_deviation                                # noqa: E402
from scripts.pothrgbd_metric_labels import DATA_DIR, load_polygons, timestamp_key  # noqa: E402
from scripts.train_relief_depth import relief_target                            # noqa: E402

OUT = os.path.join(PROJECT_DIR, "ml_results", "figures", "counterexamples")
RES = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd")
BANDS = ["Shallow", "Moderate", "Deep"]


def band(mm):
    return BANDS[0] if mm < 25 else BANDS[1] if mm < 50 else BANDS[2]


def frame(key):
    img = glob.glob(os.path.join(DATA_DIR, "images", f"{key}*.jpg"))[0]
    dep = np.load(glob.glob(os.path.join(DATA_DIR, "depths", f"{key}*.npy"))[0]).astype(np.float64)
    bgr = cv2.imread(img)
    if bgr.shape[:2] != dep.shape:
        bgr = cv2.resize(bgr, dep.shape[::-1], interpolation=cv2.INTER_AREA)
    polys = load_polygons(glob.glob(os.path.join(DATA_DIR, "labels", f"{key}*.txt"))[0], *dep.shape)
    return bgr, dep, polys


def ring_relief(values, mask):
    """Any map read the label's way: deviation from a plane fitted on the ring of road."""
    res = road_plane_deviation(values.astype(np.float64), mask, np.isfinite(values))
    return None if res is None else res[0]


def bowl(rel, mask):
    return float(np.percentile(rel[mask > 0], 90))


def main():
    import torch
    from eval_relief_pothole600 import base_model                               # noqa: E402
    from shape_from_shading import reconstruct_depth_sfs                         # noqa: E402
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nets = [R.build_model(w).to(dev).eval() for w in R.installed_weights()]
    da = base_model()

    t = pd.read_csv(os.path.join(RES, "relief_eval_potholes_v4v3.csv"))
    t = t[(t.split == "test") & (t["mask"] == "human")][["key", "pothole", "gt_mm", "relief", "regressor"]]
    tm = pd.read_csv(os.path.join(RES, "trainset_metric.csv"))
    tm = tm[tm.mask_source == "gt"][["key", "pothole", "mf_p90_mm", "cv_mean_curvature", "sh_log_area_px"]]
    tm = tm.drop_duplicates(["key", "pothole"])                       # a few potholes appear twice
    t = t.drop_duplicates(["key", "pothole"]).merge(tm, on=["key", "pothole"], how="left")

    # Old depth model and shape-from-shading, read inside each outline the label's way.
    da_b, sfs_b, sfs_rho = [], [], []
    for key, g in t.groupby("key", sort=False):
        bgr, dep, polys = frame(key)
        d = -da.infer_image(bgr, 518).astype(np.float64)              # inverse depth -> deeper = larger
        rel_true = relief_target(dep, polys, exact=True)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for p in g.pothole:
            m = polys[int(p) - 1].astype(np.uint8)
            r = ring_relief(d, m)
            da_b.append(((key, p), bowl(r, m) if r is not None else np.nan))
            h = reconstruct_depth_sfs(gray, m)
            if h is None or rel_true is None:
                sfs_b.append(((key, p), np.nan))
                sfs_rho.append(((key, p), np.nan))
                continue
            sd = 1.0 - h                                               # h = 1 is road level, 0 the bottom
            inside = (m > 0) & rel_true[1]
            sfs_b.append(((key, p), float(np.percentile(sd[m > 0], 90))))
            sfs_rho.append(((key, p), float(spearmanr(sd[inside], rel_true[0][inside]).statistic)
                            if inside.sum() > 50 else np.nan))
    for name, vals in (("da_bowl", da_b), ("sfs_bowl", sfs_b), ("sfs_rho", sfs_rho)):
        t[name] = [dict(vals)[(k, p)] for k, p in zip(t.key, t.pothole)]

    # Graph 1: rank agreement with measured depth, same 208 potholes.
    corr = {
        "Depth-Anything-V2 (old depth model)": t.da_bowl,
        "Shape-from-shading": t.sfs_bowl,
        "Boundary curvature": t.cv_mean_curvature,
        "Pothole area alone": t.sh_log_area_px,
        "MoGe-3, raw": t.mf_p90_mm,
        "MoGe-3 features + regressor": t.regressor,
        "Depth network (current)": t.relief,
    }
    corr = {k: float(spearmanr(v, t.gt_mm, nan_policy="omit").statistic) for k, v in corr.items()}
    # Graph 2: average error in mm, same potholes.
    tr = pd.read_csv(os.path.join(RES, "trainset_metric.csv"))
    median = float(tr[(tr.mask_source == "gt")].gt_mm.median())
    mae = {"Guess the median every time": float((t.gt_mm - median).abs().mean()),
           "MoGe-3, raw": float((t.mf_p90_mm - t.gt_mm).abs().mean()),
           "MoGe-3 features + regressor": float((t.regressor - t.gt_mm).abs().mean()),
           "Depth network (current)": float((t.relief - t.gt_mm).abs().mean())}
    # Graph 3: old served verdict against the measured band (742 potholes).
    l = pd.read_csv(os.path.join(RES, "legacy_served_on_pothrgbd.csv"))
    gt = pd.read_csv(os.path.join(RES, "trainset_metric.csv"))
    gt = gt[gt.mask_source == "gt"][["key", "pothole", "gt_mm"]].drop_duplicates(["key", "pothole"])
    l = l.drop_duplicates(["key", "pothole"]).merge(gt, on=["key", "pothole"], how="inner")
    l["measured"] = [band(v) for v in l.gt_mm]
    confusion = {mb: {ob: int(((l.measured == mb) & (l.legacy_served == ob)).sum()) for ob in BANDS} for mb in BANDS}

    cases = {}
    lt = l.merge(t[["key", "pothole", "relief"]], on=["key", "pothole"], how="inner")
    c = lt[(lt.gt_mm < 15) & (lt.legacy_served == "Deep")].sort_values("gt_mm")
    cases["old_too_high"] = c.iloc[0].to_dict() if len(c) else None
    c = lt[(lt.gt_mm >= 50) & (lt.semantic_verdict == "downgraded_texture_illusion")].sort_values("gt_mm", ascending=False)
    cases["dinov2_downgrade"] = c.iloc[0].to_dict() if len(c) else None
    q = t.da_bowl.rank(pct=True)
    a = t[(t.gt_mm >= 50) & (q <= 0.25)].sort_values("gt_mm", ascending=False)
    b = t[(t.gt_mm < 15) & (q >= 0.75)].sort_values("gt_mm")
    cases["da_pair"] = [a.iloc[0].to_dict(), b.iloc[0].to_dict()] if len(a) and len(b) else None
    qc = t.cv_mean_curvature.rank(pct=True)
    a = t[(qc >= 0.75) & (t.gt_mm < 15)].sort_values("gt_mm")
    b = t[(qc <= 0.25) & (t.gt_mm >= 50)].sort_values("gt_mm", ascending=False)
    cases["curv_pair"] = [a.iloc[0].to_dict(), b.iloc[0].to_dict()] if len(a) and len(b) else None
    c = t[t.gt_mm >= 10].assign(over=lambda x: x.mf_p90_mm - x.gt_mm).sort_values("over", ascending=False)
    cases["moge_raw"] = c.iloc[0].to_dict()
    c = t[(t.gt_mm >= 30)].dropna(subset=["sfs_rho"]).sort_values("sfs_rho")
    cases["sfs"] = c.iloc[0].to_dict()

    stats = {
        "n_test": int(len(t)), "n_legacy": int(len(l)),
        "old_said_deep": int((l.legacy_served == "Deep").sum()), "measured_deep": int((l.measured == "Deep").sum()),
        "old_deep_on_shallow": int(((l.legacy_served == "Deep") & (l.measured == "Shallow")).sum()),
        "measured_shallow": int((l.measured == "Shallow").sum()),
        "downgraded_measured_deep": int(((l.semantic_verdict == "downgraded_texture_illusion") & (l.measured == "Deep")).sum()),
        "downgraded_total": int((l.semantic_verdict == "downgraded_texture_illusion").sum()),
        "sfs_negative_share": float((t.sfs_rho.dropna() < 0).mean()), "sfs_rho_median": float(t.sfs_rho.median()),
        "moge_raw_over_share": float((t.mf_p90_mm > t.gt_mm).mean()),
        "moge_raw_mean_bias": float((t.mf_p90_mm - t.gt_mm).mean()),
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"spearman_vs_measured": corr, "mae_mm": mae, "old_verdict_vs_measured": confusion,
                   "stats": stats, "cases": cases, "median_guess_mm": median}, f, indent=2, default=float)
    t.to_csv(os.path.join(OUT, "test_potholes_all_methods.csv"), index=False)
    print(json.dumps({"spearman": corr, "mae": mae, "confusion": confusion, "stats": stats}, indent=1, default=float))
    for k, v in cases.items():
        rows = v if isinstance(v, list) else [v]
        print(k, [(r["key"], int(r["pothole"]), round(r["gt_mm"], 1)) for r in rows if r])
    draw(cases, nets, da)


def crop_box(mask, h, w, grow=1.8):
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    s = max(ys.max() - ys.min(), xs.max() - xs.min()) * grow / 2 + 20
    return int(max(0, cy - s)), int(min(h, cy + s)), int(max(0, cx - s)), int(min(w, cx + s))


def draw(cases, nets, da):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shape_from_shading import reconstruct_depth_sfs
    from scripts.build_metric_trainset import moge_for
    plt.rcParams.update({"font.size": 12, "figure.facecolor": "#fbfbf8", "axes.facecolor": "#fbfbf8"})
    INK, ACC = "#1f2933", "#c2410c"

    def panels(row, axes, extras):
        key, p = row["key"], int(row["pothole"])
        bgr, dep, polys = frame(key)
        m = polys[p - 1].astype(np.uint8)
        y0, y1, x0, x1 = crop_box(m, *m.shape)
        rel = relief_target(dep, polys, exact=True)
        meas = np.where(rel[1], rel[0], np.nan)
        pred = R.predict_relief(bgr, nets)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        photo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        cv2.drawContours(photo, cnt, -1, (255, 220, 0), 2)
        ims = [("Photo, outline in yellow", photo[y0:y1, x0:x1], None)] + extras(bgr, m, y0, y1, x0, x1) + [
            (f"Measured depth below road\n{row['gt_mm']:.0f} mm ({band(row['gt_mm'])})", meas[y0:y1, x0:x1], "mm"),
            (f"Current model\n{row['relief']:.0f} mm ({band(row['relief'])})", pred[y0:y1, x0:x1], "mm")]
        for ax, (title, im, kind) in zip(axes, ims):
            if kind is None:
                ax.imshow(im)
            elif kind == "mm":
                hi = max(20.0, float(np.nanpercentile(meas[y0:y1, x0:x1], 99)))
                ax.imshow(im, cmap="magma", vmin=-5, vmax=hi)
            else:
                ax.imshow(im, cmap="magma")
            ax.set_title(title, fontsize=12, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])

    def save(fig, name, headline):
        fig.suptitle(headline, fontsize=15, fontweight="bold", color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(OUT, name), dpi=110)
        plt.close(fig)

    none = lambda *a: []                                                        # noqa: E731

    r = cases.get("old_too_high")
    if r:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        panels(r, ax, none)
        save(fig, "01_old_verdict_too_high.png", f"Old system: \"Deep\".  Measured: {r['gt_mm']:.0f} mm, Shallow.")

    r = cases.get("dinov2_downgrade")
    if r:
        fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
        panels(r, ax, none)
        save(fig, "02_dinov2_downgrade.png",
             f"DINOv2 check: \"flat stain\" (ratio {r['dinov2_ratio']:.2f} < 0.9), downgraded to Shallow.  Measured: {r['gt_mm']:.0f} mm, Deep.")

    def da_map(bgr, m, y0, y1, x0, x1):
        d = -da.infer_image(bgr, 518).astype(np.float64)
        r = ring_relief(d, m)
        return [("Old depth model (Depth-Anything)\nbelow-road reading, no units", r[y0:y1, x0:x1], "rel")]

    pair = cases.get("da_pair")
    if pair:
        fig, ax = plt.subplots(2, 4, figsize=(15, 8))
        for i, r in enumerate(pair):
            panels(r, ax[i], da_map)
        save(fig, "03_depth_anything_pair.png",
             f"Old depth model ranks the {pair[1]['gt_mm']:.0f} mm dip deeper than the {pair[0]['gt_mm']:.0f} mm hole")

    pair = cases.get("curv_pair")
    if pair:
        fig, ax = plt.subplots(2, 3, figsize=(13, 9.6))
        for i, r in enumerate(pair):
            panels(r, ax[i], none)
            ax[i][0].set_title(f"Photo, curvature {r['cv_mean_curvature']:.3f}", fontsize=12, color=INK)
        save(fig, "04_curvature_pair.png",
             "Sharper outline (top) is the shallow one; the smoother outline (bottom) is the deep one")

    r = cases.get("moge_raw")
    if r:
        def moge_map(bgr, m, y0, y1, x0, x1):
            g = moge_for(r["key"], bgr, None)
            res = road_plane_deviation(g["depth"].astype(np.float64) * 1000.0, m, g["mask"] > 0)
            return [(f"MoGe-3 raw, below-road reading\n{r['mf_p90_mm']:.0f} mm", res[0][y0:y1, x0:x1], "rel")]
        fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
        panels(r, ax, moge_map)
        save(fig, "05_moge_raw.png", f"MoGe-3 reads {r['mf_p90_mm']:.0f} mm; measured {r['gt_mm']:.0f} mm")

    r = cases.get("sfs")
    if r:
        def sfs_map(bgr, m, y0, y1, x0, x1):
            h = reconstruct_depth_sfs(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), m)
            sd = np.where(m > 0, 1.0 - h, np.nan)
            return [(f"Shape-from-shading depth\nrank agreement with measured {r['sfs_rho']:+.2f}", sd[y0:y1, x0:x1], "rel")]
        fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
        panels(r, ax, sfs_map)
        save(fig, "06_shape_from_shading.png", "Shape-from-shading puts the deep part where the road is high, and the reverse")
    print("figures in", os.path.relpath(OUT, PROJECT_DIR))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
    main()
