"""
Merge the team's labels for one annotation pack: agreement, disputes, final labels.

    python scripts/merge_annotations.py packs/water_v1 returns/labels_water_v1_*.csv
    python scripts/merge_annotations.py packs/water_v1 returns/*.csv --consensus returns/labels_water_v1_consensus.csv

Reads the returned labels_<task>_v<N>_<name>.csv files (from any folder) and:

  1. checks each file belongs to this pack version and names an assigned annotator;
  2. measures agreement on every overlapping pair: percent agreement and Cohen's kappa
     (agreement corrected for chance: 1 = perfect, 0 = no better than guessing);
  3. an item is FINAL when its labellers agree; otherwise it is DISPUTED and goes to
     disputed.csv for the joint review (annotate_pack.py --review disputed.csv
     --annotator consensus). A consensus file, when given, settles disputes;
  4. writes the label file the evaluation script already reads, with the frozen
     outline attached so evaluation scores exactly the pothole people saw:
        water     -> data/pothole_water_labels.csv   (scripts/eval_pothole_water.py)
        illusion  -> data/illusion_labels.csv        (scripts/calibrate_semantic.py)
  5. saves agreement_report.txt in the pack folder.

"unclear" and "not_a_pothole" are kept as answers: when both labellers give one, the item
is final but left out of evaluation; one of them against a real label is a dispute. The
not_a_pothole share is reported: it is the segmenter's false-detection rate on this sample.
"""
import argparse
import csv
import glob
import itertools
import os
import sys
from collections import Counter, defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.annotate_pack import load_pack, read_csv, write_csv    # noqa: E402

OUTPUTS = {
    "water": (os.path.join(PROJECT_DIR, "data", "pothole_water_labels.csv"),
              ["image", "pothole_idx", "source", "label", "polygon"]),
    "illusion": (os.path.join(PROJECT_DIR, "data", "illusion_labels.csv"),
                 ["image", "pothole_idx", "label", "stratum", "interior_lapvar", "interior_mean", "polygon"]),
}


EXCLUDED = {"unclear", "not_a_pothole"}          # final, but never scored


def cohen_kappa(a, b):
    """Cohen's kappa for two raters over the same items."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(a) | set(b)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack")
    ap.add_argument("labels", nargs="*", help="returned label files (default: labels_*.csv in the pack folder)")
    ap.add_argument("--consensus", default=None, help="labels file from the joint review")
    ap.add_argument("--dry-run", action="store_true", help="report only; do not write data/ files")
    args = ap.parse_args()

    task, manifest = load_pack(args.pack)
    items = {r["item_id"]: r for r in manifest}
    files = []
    for pattern in args.labels or [os.path.join(args.pack, "labels_*.csv")]:
        files += glob.glob(pattern)
    files = sorted({f for f in files if not f.endswith("_consensus.csv")})

    labels = defaultdict(dict)                   # item_id -> {annotator: label}
    problems = []
    for f in files:
        for r in read_csv(f):
            if str(r.get("pack_version")) != str(task["version"]):
                problems.append(f"{os.path.basename(f)}: pack version {r.get('pack_version')} != {task['version']}")
                break
            iid, who = r["item_id"], r["annotator"]
            if iid not in items or who not in items[iid]["assigned_to"].split(";"):
                problems.append(f"{os.path.basename(f)}: {iid} was not assigned to {who}")
                continue
            if who in labels[iid] and labels[iid][who] != r["label"]:
                problems.append(f"{os.path.basename(f)}: {who} gave {iid} two different answers; keeping the last")
            labels[iid][who] = r["label"]
    consensus = {r["item_id"]: r["label"] for r in read_csv(args.consensus)} if args.consensus else {}

    lines = [f"{task['title']}  (pack v{task['version']}, {len(manifest)} potholes)", ""]
    progress = {a: sum(a in labels[i] for i in items if a in items[i]["assigned_to"].split(";"))
                for a in task["annotators"]}
    assigned = {a: sum(a in r["assigned_to"].split(";") for r in manifest) for a in task["annotators"]}
    lines.append("Returned:  " + "   ".join(f"{a} {progress[a]}/{assigned[a]}" for a in task["annotators"]))

    lines += ["", f"{'pair':28s} {'items':>5s} {'agree':>6s} {'kappa':>6s}"]
    all_a, all_b = [], []
    for a, b in itertools.combinations(task["annotators"], 2):
        both = [i for i in items if a in labels[i] and b in labels[i]]
        la, lb = [labels[i][a] for i in both], [labels[i][b] for i in both]
        all_a += la
        all_b += lb
        if both:
            agree = sum(x == y for x, y in zip(la, lb)) / len(both)
            lines.append(f"{a + ' & ' + b:28s} {len(both):5d} {agree:6.1%} {cohen_kappa(la, lb):6.2f}")
    if all_a:
        lines.append(f"{'all pairs':28s} {len(all_a):5d} "
                     f"{sum(x == y for x, y in zip(all_a, all_b)) / len(all_a):6.1%} {cohen_kappa(all_a, all_b):6.2f}")

    final, disputed, pending = {}, [], []
    for iid, r in items.items():
        need = r["assigned_to"].split(";")
        got = labels[iid]
        if iid in consensus:
            final[iid] = consensus[iid]
        elif len([a for a in need if a in got]) < len(need):
            pending.append(iid)
        elif len(set(got[a] for a in need)) == 1:
            final[iid] = got[need[0]]
        else:
            row = {"item_id": iid, "source": r["source"]}
            for n, a in enumerate(need, start=1):
                row[f"annotator_{n}"], row[f"label_{n}"] = a, got[a]
            disputed.append(row)

    by_source = defaultdict(lambda: [0, 0])
    for iid in items:
        if iid in final or any(d["item_id"] == iid for d in disputed):
            by_source[items[iid]["source"]][0] += iid in final and iid not in consensus
            by_source[items[iid]["source"]][1] += 1
    lines += ["", "Agreed without review, by source:  " +
              "   ".join(f"{s} {v[0]}/{v[1]}" for s, v in sorted(by_source.items()))]
    lines += ["", f"final {len(final)}   disputed {len(disputed)}   not yet labelled by both {len(pending)}",
              "final labels: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(final.values()).items()))]
    if problems:
        lines += ["", "Problems:"] + [f"  - {p}" for p in problems]

    disputed_path = os.path.join(args.pack, "disputed.csv")
    if disputed:
        if not args.dry_run:
            write_csv(disputed_path, disputed, list(disputed[0].keys()))
        lines += ["", f"{len(disputed)} disputed items -> {disputed_path}",
                  "  joint review:  python annotate_pack.py --annotator consensus --review disputed.csv"]

    out_path, fields = OUTPUTS[task["task"]]
    usable = [iid for iid, lab in final.items() if lab not in EXCLUDED]
    n_np = sum(lab == "not_a_pothole" for lab in final.values())
    if final:
        lines += [f"segmenter false detections (both said not_a_pothole): {n_np} of {len(final)} "
                  f"({n_np / len(final):.0%})"]
    if not args.dry_run and usable:
        rows = []
        for iid in usable:
            r = items[iid]
            row = {"image": r["image"], "pothole_idx": r["pothole_idx"], "source": r["source"],
                   "label": final[iid], "polygon": r["polygon"], "stratum": r.get("stratum", ""),
                   "interior_lapvar": r.get("interior_lapvar", ""), "interior_mean": r.get("interior_mean", "")}
            if task["task"] == "illusion":
                row["image"] = os.path.basename(r["image"])     # calibrate_semantic finds images by name
            rows.append({k: row[k] for k in fields})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_csv(out_path, rows, fields)
        lines += ["", f"wrote {len(rows)} final labels (unclear / not_a_pothole excluded) -> "
                      f"{os.path.relpath(out_path, PROJECT_DIR)}"]

    report = "\n".join(lines)
    print(report)
    if not args.dry_run:
        with open(os.path.join(args.pack, "agreement_report.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
