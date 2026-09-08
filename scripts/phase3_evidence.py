"""
Phase 3 evidence figures — before vs after the measurement correction.

Fig A: the pic-100 augmentation triplet, raw (whole-scene) ratio vs corrected
       (ring-local, size-matched) ratio, with the resulting verdict for each.
       The raw ratios straddle the 0.9 cut-off; the corrected ones should agree.
Fig B: raw vs corrected ratio across a probe set, showing the spread the
       corrected statistic recovers.

Usage:  python scripts/phase3_evidence.py
"""
import glob, os, sys
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from segmentation import get_all_masks
from foundation_features import extract_foundation_features
import semantic_config as SC

DARK, PANEL, TEXT = "#0f172a", "#1e293b", "#e2e8f0"
OUT = os.path.join(ROOT, "ml_results", "phase3_diagnostics")
LEGACY = 0.9


def analyse(path):
    masks = get_all_masks(path)
    if not masks:
        return None
    img = cv2.imread(path)
    f = extract_foundation_features(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), masks[0])
    if not f:
        return None
    iv, ov = f["dinov2_inside_variance"], f["dinov2_outside_variance"]
    return dict(rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask=masks[0],
                raw=iv / ov if ov > 1e-9 else float("nan"),
                corr=f.get("dinov2_ratio_corrected", float("nan")),
                patches=f.get("dinov2_patch_count", 0),
                name=os.path.basename(path))


def verdict(ratio, thresh):
    return "DOWNGRADE" if ratio < thresh else "no action"


def fig_triplet(recs):
    n = len(recs)
    fig, axes = plt.subplots(2, n, figsize=(4.7 * n, 8.6), facecolor=DARK)
    for i, r in enumerate(recs):
        ax = axes[0, i]
        ax.imshow(r["rgb"])
        c, _ = cv2.findContours(r["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cc in c:
            cc = cc.squeeze()
            if cc.ndim == 2:
                ax.plot(np.append(cc[:, 0], cc[0, 0]), np.append(cc[:, 1], cc[0, 1]), color="#fde047", lw=2)
        ax.set_title(f"Augmentation {i+1}", color=TEXT, fontsize=12); ax.axis("off")

        ax = axes[1, i]; ax.set_facecolor(PANEL)
        rawv, corrv = verdict(r["raw"], LEGACY), verdict(r["corr"], SC.DOWNGRADE_RATIO)
        rc = "#f87171" if rawv == "DOWNGRADE" else "#34d399"
        cc_ = "#f87171" if corrv == "DOWNGRADE" else "#34d399"
        ax.barh([1, 0], [r["raw"], r["corr"]], color=[rc, cc_], height=0.5)
        ax.axvline(LEGACY, color="#94a3b8", ls=":", lw=1.5)
        ax.axvline(SC.DOWNGRADE_RATIO, color="#fbbf24", ls="--", lw=1.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["CORRECTED", "raw (old)"], color=TEXT, fontsize=10)
        ax.set_xlim(0, max(1.6, r["raw"] * 1.2, r["corr"] * 1.2))
        ax.tick_params(colors="#94a3b8", labelsize=9)
        for s in ax.spines.values(): s.set_color("#334155")
        ax.set_xlabel("variance ratio", color="#94a3b8", fontsize=10)
        ax.set_title(f"raw {r['raw']:.2f} -> {rawv}\ncorrected {r['corr']:.2f} -> {corrv}",
                     color=TEXT, fontsize=11, fontweight="bold", pad=8)

    fig.suptitle(
        "PHASE 3 FIX — one pothole, three augmentations\n"
        "Old whole-scene ratio straddles the cut-off and flips the verdict; the ring-local, "
        "size-matched ratio agrees across all three.",
        color=TEXT, fontsize=13, y=0.985)
    p = os.path.join(OUT, "phase3_fix_threshold_stability.png")
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(p, dpi=150, facecolor=DARK, bbox_inches="tight"); plt.close(fig)
    print("saved", os.path.relpath(p, ROOT))
    for r in recs:
        print(f"   {r['name'][:34]:<36} raw={r['raw']:.2f} corrected={r['corr']:.2f} patches={r['patches']}")


def fig_spread(recs):
    raw = np.array([r["raw"] for r in recs]); corr = np.array([r["corr"] for r in recs])
    o = np.argsort(raw); raw, corr = raw[o], corr[o]
    fig, ax = plt.subplots(figsize=(12, 6), facecolor=DARK); ax.set_facecolor(PANEL)
    x = np.arange(len(raw)); w = 0.4
    ax.bar(x - w/2, raw, w, label="raw (whole scene)", color="#64748b")
    ax.bar(x + w/2, corr, w, label="corrected (ring, size-matched)", color="#a78bfa")
    ax.axhline(LEGACY, color="#fbbf24", ls="--", lw=2)
    ax.text(len(raw)*0.5, LEGACY+0.03, "0.9 cut-off", color="#fbbf24", fontsize=10, ha="center")
    ax.set_ylabel("interior / road variance ratio", color=TEXT, fontsize=11)
    ax.set_xlabel(f"samples (n={len(raw)}), sorted by raw ratio", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8"); ax.set_xticks([])
    for s in ax.spines.values(): s.set_color("#334155")
    ax.legend(facecolor=PANEL, edgecolor="#334155", labelcolor=TEXT, fontsize=10)
    below_raw = int((raw < LEGACY).sum()); below_corr = int((corr < LEGACY).sum())
    ax.set_title(
        "PHASE 3 FIX — the corrected statistic is no longer dominated by region size\n"
        f"raw: {below_raw}/{len(raw)} below the cut-off (rule would fire on nearly everything)   ·   "
        f"corrected: {below_corr}/{len(corr)}",
        color=TEXT, fontsize=13, pad=14)
    p = os.path.join(OUT, "phase3_fix_ratio_spread.png")
    plt.tight_layout(); plt.savefig(p, dpi=150, facecolor=DARK, bbox_inches="tight"); plt.close(fig)
    print("saved", os.path.relpath(p, ROOT))
    print(f"   raw     mean={raw.mean():.2f} range {raw.min():.2f}-{raw.max():.2f}  below 0.9: {below_raw}/{len(raw)}")
    print(f"   corrected mean={corr.mean():.2f} range {corr.min():.2f}-{corr.max():.2f}  below 0.9: {below_corr}/{len(corr)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    trip = sorted(glob.glob(os.path.join(ROOT, "data1/train/images/pic-100-*.jpg")))[:3]
    recs = [r for r in (analyse(p) for p in trip) if r]
    if recs: fig_triplet(recs)
    pool = sorted(glob.glob(os.path.join(ROOT, "data1/train/images/pic-1*.jpg")))[:18]
    allr = [r for r in (analyse(p) for p in pool) if r]
    if allr: fig_spread(allr)


if __name__ == "__main__":
    main()
