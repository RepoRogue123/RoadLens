"""
Index every image in the project by origin, and find the contamination.

What this answers
-----------------
    Which dataset did this image come from?
    Does any evaluation split contain a photograph the model trained on?
    Do the same photographs appear in more than one corpus?

The second and third questions are not hypothetical. `merged_dataset` was found
to share 45 source photographs between train and valid and 49 between train and
test — 9-11% of each evaluation split had already been seen. That inflates every
metric computed on it, independently of the label circularity documented
elsewhere.

Why perceptual hashing rather than MD5
--------------------------------------
`scripts/deduplicate.py` uses MD5, which detects only byte-identical files. That
misses the case that actually matters here: the same photograph re-encoded,
resized or re-exported by Roboflow under a new name. Public pothole datasets are
heavily forked and re-uploaded, so a JPEG re-save is the normal way a duplicate
propagates, and MD5 is blind to all of it.

pHash is computed the standard way: reduce to 32x32 grey, take the 2-D DCT, keep
the top-left 8x8 low-frequency block, and threshold it against its own median
(excluding the DC term, which only encodes overall brightness). The result is a
64-bit fingerprint that survives re-encoding, mild rescaling and small quality
changes. Two images are near-duplicates when the Hamming distance between their
hashes is small.

Two levels of identity, and both are needed
-------------------------------------------
    source_stem   filename identity with Roboflow's augmentation suffix removed.
                  Catches augmented variants of one photo — the leak that is
                  invisible if you group by filename.
    phash         content identity. Catches the same photo arriving from a
                  different corpus under a different name.

Outputs to ml_results/provenance/:
    image_index.csv           one row per image: source, split, stem, phash
    contamination_report.txt  split leakage and cross-source duplicates

Usage:
    python scripts/build_provenance_index.py
    python scripts/build_provenance_index.py --max-distance 6
"""
import argparse
import csv
import glob
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from dataset_registry import (                                  # noqa: E402
    REGISTRY, source_stem, merged_origin, pothrgbd_key,
)

OUT_DIR = os.path.join(PROJECT_DIR, "ml_results", "provenance")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

# Hamming distance below which two 64-bit pHashes are treated as the same photo.
# 0 is a re-encode; up to ~8 tolerates rescaling and quality loss; beyond ~12 the
# match becomes "similar scene" rather than "same photo" and produces noise.
DEFAULT_MAX_DISTANCE = 6

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(path: str) -> int:
    """64-bit perceptual hash, or -1 if the image cannot be read."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1
    im = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(im)[:8, :8]
    # Exclude the DC coefficient from the median: it carries overall brightness,
    # not structure, and including it makes the hash sensitive to exposure.
    med = float(np.median(np.delete(d.flatten(), 0)))
    bits = (d > med).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 hash arrays."""
    x = np.bitwise_xor(a[:, None], b[None, :]).astype(np.uint64)
    return _POPCOUNT[x.view(np.uint8).reshape(*x.shape, 8)].sum(axis=-1)


def split_of(path: str, source) -> str:
    rel = os.path.relpath(path, source.path).replace("\\", "/")
    for part in rel.split("/"):
        if part in ("train", "valid", "val", "test"):
            return part
    return "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE)
    ap.add_argument("--limit-per-source", type=int, default=0)
    ap.add_argument("--from-index", action="store_true",
                    help="Rebuild the report from an existing image_index.csv "
                         "without re-hashing 9700 images")
    args = ap.parse_args()

    rows = []
    idx_path = os.path.join(OUT_DIR, "image_index.csv")

    if args.from_index and os.path.isfile(idx_path):
        with open(idx_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["phash"] = int(r["phash"])
        print(f"Loaded {len(rows)} rows from {os.path.relpath(idx_path, PROJECT_DIR)}\n")

    print("Indexing image sources\n" if not rows else "")
    for src in (REGISTRY if not rows else []):
        if not src.present:
            print(f"  [absent ] {src.id}")
            continue
        files = [f for f in glob.glob(os.path.join(src.path, "**", "*"),
                                      recursive=True)
                 if f.lower().endswith(IMG_EXT)]
        files.sort()
        if args.limit_per_source:
            files = files[:args.limit_per_source]

        for i, f in enumerate(files, start=1):
            h = phash(f)
            if h < 0:
                continue
            rows.append({
                "source": src.id,
                "split": split_of(f, src),
                "path": os.path.relpath(f, PROJECT_DIR).replace("\\", "/"),
                "filename": os.path.basename(f),
                "source_stem": source_stem(f),
                "origin": merged_origin(f) if src.key_rule == "prefix" else src.id,
                "frame_key": pothrgbd_key(f) or "",
                "phash": h,
                "region": src.region,
                "licence": src.licence,
            })
            if i % 500 == 0:
                print(f"    {src.id}: {i}/{len(files)}", flush=True)
        print(f"  [indexed] {src.id:16s} {len(files):5d} images")

    if not rows:
        print("\nNothing indexed.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    idx = idx_path
    if not args.from_index:
        with open(idx, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # ── report ──
    lines = []

    def out(s=""):
        """
        Print encoding-safe, write faithfully.

        The Windows console is cp1252 and raises UnicodeEncodeError on any
        character outside it — a crash mid-report, after the analysis has
        already been done. The report FILE is UTF-8 and keeps the real text;
        only the console copy is degraded.
        """
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "ascii"
            print(s.encode(enc, errors="replace").decode(enc, errors="replace"))

    out()
    out("=" * 74)
    out("PROVENANCE AND CONTAMINATION REPORT")
    out("=" * 74)

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)

    out(f"\n{len(rows)} images across {len(by_src)} sources\n")
    out(f"  {'source':17s}{'files':>7s}{'photos':>9s}{'aug x':>7s}  region")
    out("  " + "-" * 70)
    for sid, rs in sorted(by_src.items()):
        stems = {r["source_stem"] for r in rs}
        out(f"  {sid:17s}{len(rs):>7d}{len(stems):>9d}"
            f"{len(rs)/max(len(stems),1):>7.2f}  {rs[0]['region'][:34]}")

    # 1. split leakage, by source photograph
    out("\n\n1. SPLIT LEAKAGE  (same photograph on both sides of a split)\n")
    any_leak = False
    for sid, rs in sorted(by_src.items()):
        splits = defaultdict(set)
        for r in rs:
            if r["split"] != "-":
                splits[r["split"]].add(r["source_stem"])
        names = sorted(splits)
        if len(names) < 2:
            out(f"  {sid:17s} single split — not applicable")
            continue
        worst = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                ov = splits[a] & splits[b]
                if ov:
                    any_leak = True
                    worst.append((a, b, len(ov), len(splits[b])))
        if worst:
            out(f"  {sid:17s} *** CONTAMINATED ***")
            for a, b, n, tot in worst:
                out(f"{'':19s}{a} ∩ {b}: {n} photographs "
                    f"({100*n/max(tot,1):.1f}% of {b})")
        else:
            out(f"  {sid:17s} clean — no photograph spans a split")
    if any_leak:
        out("\n  Any metric computed on a contaminated split is inflated.")
        out("  Re-split grouping on `source_stem` before trusting it.")

    # 2. augmentation groups
    out("\n\n2. AUGMENTATION GROUPS  (one photograph, several files)\n")
    for sid, rs in sorted(by_src.items()):
        groups = defaultdict(int)
        for r in rs:
            groups[r["source_stem"]] += 1
        multi = {k: v for k, v in groups.items() if v > 1}
        if multi:
            out(f"  {sid:17s} {len(multi)} photographs have multiple variants, "
                f"max {max(multi.values())}")
            out(f"{'':19s}group by source_stem in any split or CV fold")
        else:
            out(f"  {sid:17s} one file per photograph")

    # 3. cross-source duplicates, by content
    out("\n\n3. CROSS-SOURCE DUPLICATES  "
        f"(pHash Hamming <= {args.max_distance})\n")
    hashes = np.array([r["phash"] for r in rows], dtype=np.uint64)
    srcs = np.array([r["source"] for r in rows])
    uniq = sorted(set(srcs))
    found = 0
    for i in range(len(uniq)):
        for j in range(i + 1, len(uniq)):
            ia = np.flatnonzero(srcs == uniq[i])
            ib = np.flatnonzero(srcs == uniq[j])
            hits = 0
            examples = []
            step = 400
            for k in range(0, len(ia), step):
                blk = ia[k:k + step]
                D = hamming_matrix(hashes[blk], hashes[ib])
                m = D <= args.max_distance
                hits += int(m.any(axis=1).sum())
                if m.any() and len(examples) < 3:
                    r0, c0 = np.argwhere(m)[0]
                    examples.append((rows[blk[r0]]["filename"],
                                     rows[ib[c0]]["filename"]))
            if hits:
                found += 1
                out(f"  {uniq[i]} <-> {uniq[j]}: {hits} matching photographs")
                for a, b in examples:
                    out(f"{'':6s}{a[:44]}  ~  {b[:44]}")
    if not found:
        out("  none — the corpora are genuinely disjoint in content")

    out("\n\n4. LICENCES IN PLAY\n")
    for sid, rs in sorted(by_src.items()):
        out(f"  {sid:17s} {rs[0]['licence']}")

    out()
    out("=" * 74)

    rep = os.path.join(OUT_DIR, "contamination_report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  wrote {os.path.relpath(idx, PROJECT_DIR)}")
    print(f"  wrote {os.path.relpath(rep, PROJECT_DIR)}")


if __name__ == "__main__":
    main()
