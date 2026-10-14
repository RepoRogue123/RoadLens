"""
Freeze an annotation task into a self-contained pack for the team (coordinator only).

Why packs
---------
The labelling scripts re-run the segmenter on every photo and key each label by
"pothole number N in this image". On a machine with different weights the outlines
differ and "pothole 1" can be a different pothole — labels silently land on the wrong
object. Teammates also do not have the gitignored datasets or the ML environment.

A pack fixes both: the outline is computed ONCE here with the production segmenter and
stored as a polygon; the view each person sees is pre-rendered; and labelling needs only
opencv + numpy (scripts/annotate_pack.py, copied into the pack).

Assignment: each pothole goes to exactly `--overlap` people (default 2 of 3). Items are
shuffled within each source/stratum and dealt to annotator pairs in rotation, so every
person gets the same share of every source and every pair overlaps on the same amount —
the overlap is what lets scripts/merge_annotations.py measure agreement.

Two tasks rate HAZARD (Methods 2B and 3B of scripts/cvpr_severity_framework.py). Their
potholes come from data/hazard_pool.csv (scripts/build_hazard_pool.py: every dataset with
photos, outlines and measurements already frozen), so no segmenter runs here:
    consequence    one pothole: how bad would hitting it be, 1-5 (the risk matrix's scale)
    hazard_pairs   two potholes side by side: which is more dangerous to drive into?
                   Pairs are drawn at random from the pool, each pothole in about
                   --pair-factor pairs, never two from the same photo.

Usage:
    python scripts/build_annotation_pack.py --task water --annotators atharva,vyankatesh,shashank
    python scripts/build_annotation_pack.py --task illusion --annotators atharva,vyankatesh,shashank
    python scripts/build_annotation_pack.py --task consequence --annotators atharva,vyankatesh,shashank
    python scripts/build_annotation_pack.py --task hazard_pairs --annotators atharva,vyankatesh,shashank

Output: packs/<task>_v<N>/ and packs/RoadLens_<task>_pack_v<N>.zip (packs/ is gitignored).
"""
import argparse
import itertools
import json
import os
import shutil
import sys
from datetime import date

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import segmentation                                                     # noqa: E402
from scripts.annotate_pack import mask_from_polygon, polygon_from_mask, write_csv   # noqa: E402

PACKS = os.path.join(PROJECT_DIR, "packs")
SEED = 20260929

TASKS = {
    "water": {
        "title": "RoadLens - does this pothole hold water?",
        "question": "Does this pothole hold water?",
        "keys": {"w": "water", "d": "dry", "3": "unclear", "x": "not_a_pothole"},
        "definitions": {
            "water": "Any standing water inside the yellow outline, even a thin film or a small pool.",
            "dry": "No standing water. Damp-looking soil, dark asphalt or wet-looking shadows are dry.",
            "unclear": "You genuinely cannot tell. Never guess - unclear is a valid answer.",
            "not_a_pothole": "The yellow outline is not around a pothole (sky, trees, a car, a logo, a thin edge sliver) or misses the pothole. Excluded from evaluation; it tells us how often the segmenter is wrong.",
        },
    },
    "illusion": {
        "title": "RoadLens - real crater or texture illusion?",
        "question": "Real crater or flat texture illusion?",
        "keys": {"1": "real_crater", "2": "texture_illusion", "3": "unclear", "x": "not_a_pothole"},
        "definitions": {
            "real_crater": "The surface is genuinely broken: missing material, rubble, a cavity with depth.",
            "texture_illusion": "Flat road that only looks like a hole: a stain, patch, shadow or dark repair.",
            "unclear": "You genuinely cannot tell. Never guess - unclear is a valid answer.",
            "not_a_pothole": "The yellow outline is not around a pothole (sky, trees, a car, a logo, a thin edge sliver) or misses the pothole. Excluded from evaluation; it tells us how often the segmenter is wrong.",
        },
    },
    "consequence": {
        "title": "RoadLens - how bad would it be to hit this pothole?",
        "question": "If a car, motorbike or bicycle hit this at normal speed: 1 insignificant ... 5 major",
        "keys": {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "0": "unclear", "x": "not_a_pothole"},
        "definitions": {
            "1": "Insignificant: a bump nobody would notice.",
            "2": "Minor: uncomfortable jolt; no damage expected.",
            "3": "Moderate: could damage a tyre or rim, or unsettle a two-wheeler.",
            "4": "Significant: likely tyre/wheel damage; a two-wheeler could lose control.",
            "5": "Major: could cause a crash or serious vehicle damage.",
            "unclear": "You cannot judge it from the photo (blurred, too far, hidden). Never guess.",
            "not_a_pothole": "The yellow outline is not around a pothole. Excluded.",
        },
    },
    "hazard_pairs": {
        "title": "RoadLens - which pothole is more dangerous?",
        "question": "Which pothole is more dangerous to drive into: A (left) or B (right)?",
        "keys": {"a": "a", "b": "b", "e": "equal", "0": "unclear", "x": "not_a_pothole"},
        "definitions": {
            "a": "A (left) is more dangerous to drive into.",
            "b": "B (right) is more dangerous to drive into.",
            "equal": "About equally dangerous.",
            "unclear": "You cannot judge one of them from the photo. Never guess.",
            "not_a_pothole": "At least one yellow outline is not around a pothole. Excluded.",
        },
    },
}
POOL_TASKS = ("consequence", "hazard_pairs")


def candidates(task):
    """(image_path, pothole_idx, source, stratum, extras) from the existing selectors."""
    if task == "water":
        import scripts.label_pothole_water as L
        return [(os.path.join(PROJECT_DIR, c["image"]), int(c["pothole_idx"]), c["source"], c["source"], {})
                for c in L.build_candidates()]
    import scripts.label_illusions as L
    out = []
    for path, stratum in L.build_candidates(120, True):
        top = os.path.relpath(path, PROJECT_DIR).replace("\\", "/").split("/")[0]   # data1 / merged_dataset
        out.append((path, 0, "kaggle" if top == "data1" else top, stratum, {}))
    return out


def pool_items(task, pair_factor):
    """Items for the hazard tasks, from data/hazard_pool.csv (outlines already frozen)."""
    import pandas as pd
    from scripts.build_hazard_pool import POOL_CSV
    if not os.path.isfile(POOL_CSV):
        sys.exit("data/hazard_pool.csv missing - run scripts/build_hazard_pool.py first")
    pool = pd.read_csv(POOL_CSV).to_dict("records")
    if task == "consequence":
        return [{"pool_ids": [r["pool_id"]], "source": r["source"], "stratum": r["source"]} for r in pool]
    rng = np.random.default_rng(SEED)
    n_pairs = int(round(len(pool) * pair_factor / 2))
    seen, out = set(), []
    while len(out) < n_pairs:
        i, j = rng.choice(len(pool), 2, replace=False)
        a, b = pool[i], pool[j]
        key = tuple(sorted((a["pool_id"], b["pool_id"])))
        if a["image"] == b["image"] or key in seen:
            continue
        seen.add(key)
        out.append({"pool_ids": [a["pool_id"], b["pool_id"]], "source": f"{a['source']}|{b['source']}",
                    "stratum": "pairs"})
    return out


def pool_view(rows, caption):
    """One pothole (photo + zoomed crop), or two side by side labelled A and B."""
    import scripts.label_pothole_water as L
    panels = []
    for k, r in enumerate(rows):
        img = cv2.imread(os.path.join(PROJECT_DIR, r["image"]))
        mask = mask_from_polygon(r["polygon"], *img.shape[:2])
        v = L.build_view(img, mask, "")[:-60]
        if len(rows) == 2:
            v = cv2.resize(v, (int(v.shape[1] * 360 / v.shape[0]), 360))
            cv2.putText(v, "AB"[k], (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 3)
        panels.append(v)
    if len(panels) == 2:
        h = max(p.shape[0] for p in panels)
        gap = np.full((h, 16, 3), 255, np.uint8)
        view = np.hstack([panels[0], gap, panels[1]])
    else:
        view = panels[0]
    bar = np.zeros((40, view.shape[1], 3), np.uint8)
    cv2.putText(bar, caption, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([view, bar])


def view_for(task, img, mask, caption):
    if task == "water":
        import scripts.label_pothole_water as L
        v = L.build_view(img, mask, caption)
    else:
        import scripts.label_illusions as L
        v = L.build_view(img, mask, caption, "")
    return v[:-84] if task == "illusion" else v[:-60]          # drop the tool's own key legend bar


def assign(items, annotators, overlap):
    """Deal items to annotator groups in rotation, separately within each stratum."""
    groups = list(itertools.combinations(annotators, overlap))
    rng = np.random.default_rng(SEED)
    by_stratum = {}
    for it in items:
        by_stratum.setdefault(it["stratum"], []).append(it)
    k = 0
    for s in sorted(by_stratum):
        bucket = by_stratum[s]
        for j in rng.permutation(len(bucket)):
            bucket[j]["assigned_to"] = ";".join(groups[k % len(groups)])
            k += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--annotators", required=True, help="comma-separated first names")
    ap.add_argument("--overlap", type=int, default=2)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--pair-factor", type=float, default=3.0, help="hazard_pairs: pairs each pothole appears in")
    args = ap.parse_args()
    annotators = [a.strip().lower() for a in args.annotators.split(",") if a.strip()]
    if not 1 <= args.overlap <= len(annotators):
        sys.exit("--overlap must be between 1 and the number of annotators")

    spec = TASKS[args.task]
    out = os.path.join(PACKS, f"{args.task}_v{args.version}")
    if os.path.isdir(out):
        sys.exit(f"{os.path.relpath(out, PROJECT_DIR)} exists. Packs are immutable once shared - "
                 f"use --version {args.version + 1}.")
    os.makedirs(os.path.join(out, "items"))

    items = []
    if args.task in POOL_TASKS:
        import pandas as pd
        from scripts.build_hazard_pool import POOL_CSV
        pool = {r["pool_id"]: r for r in pd.read_csv(POOL_CSV).to_dict("records")}
        for n, it in enumerate(pool_items(args.task, args.pair_factor), start=1):
            item_id = f"{args.task[0]}{n:04d}"
            cv2.imwrite(os.path.join(out, "items", f"{item_id}.jpg"),
                        pool_view([pool[i] for i in it["pool_ids"]], item_id), [cv2.IMWRITE_JPEG_QUALITY, 88])
            items.append({"item_id": item_id, "source": it["source"], "stratum": it["stratum"],
                          "pool_ids": ";".join(it["pool_ids"])})
    for n, (path, idx, source, stratum, _extra) in enumerate(
            [] if args.task in POOL_TASKS else candidates(args.task), start=1):
        img = cv2.imread(path)
        masks = segmentation.get_all_masks(path)
        if img is None or idx >= len(masks):
            continue
        poly = polygon_from_mask(masks[idx])
        if poly is None:
            continue
        frozen = mask_from_polygon(poly, *img.shape[:2])        # show exactly what is stored
        item_id = f"{args.task[0]}{n:04d}"
        cv2.imwrite(os.path.join(out, "items", f"{item_id}.jpg"),
                    view_for(args.task, img, frozen, f"{item_id}  ({source})"), [cv2.IMWRITE_JPEG_QUALITY, 88])
        row = {"item_id": item_id, "source": source, "stratum": stratum,
               "image": os.path.relpath(path, PROJECT_DIR).replace("\\", "/"),
               "pothole_idx": idx, "polygon": json.dumps(poly, separators=(",", ":"))}
        if args.task == "illusion":
            import scripts.label_illusions as L
            st = L.interior_stats(img, frozen) or (float("nan"), float("nan"))
            row["interior_lapvar"], row["interior_mean"] = round(st[0], 2), round(st[1], 2)
        items.append(row)

    assign(items, annotators, args.overlap)
    fields = list(items[0].keys())
    write_csv(os.path.join(out, "manifest.csv"), items, fields)
    task_json = {"task": args.task, "version": args.version, "created": str(date.today()),
                 "annotators": annotators, "overlap": args.overlap,
                 "segmenter": "data/hazard_pool.csv" if args.task in POOL_TASKS else os.path.basename(segmentation.MODEL_PATH),
                 "segmenter_conf": segmentation.CONF_THRESHOLD, **spec}
    with open(os.path.join(out, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task_json, f, indent=2)
    shutil.copy2(os.path.join(PROJECT_DIR, "scripts", "annotate_pack.py"), out)
    with open(os.path.join(out, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"{spec['title']}  (pack v{args.version})\n\n"
                "1. pip install opencv-python numpy\n"
                "2. In this folder run:   python annotate_pack.py --annotator <your first name>\n"
                f"   names: {', '.join(annotators)}\n"
                f"3. Send back the file labels_{args.task}_v{args.version}_<your name>.csv\n\n"
                + "\n".join(f"  {k} = {v}: {spec['definitions'][v]}" for k, v in spec["keys"].items()) + "\n")

    zip_path = shutil.make_archive(os.path.join(PACKS, f"RoadLens_{args.task}_pack_v{args.version}"),
                                   "zip", root_dir=PACKS, base_dir=os.path.basename(out))
    per = {a: sum(a in r["assigned_to"].split(";") for r in items) for a in annotators}
    what = "pairs" if args.task == "hazard_pairs" else "potholes"
    print(f"{len(items)} {what} frozen" + (" from data/hazard_pool.csv" if args.task in POOL_TASKS else
                                          f" with {task_json['segmenter']} @ {task_json['segmenter_conf']}"))
    print(f"  each labelled by {args.overlap}; per person: {per}")
    print(f"  wrote {os.path.relpath(out, PROJECT_DIR)}/ and {os.path.relpath(zip_path, PROJECT_DIR)} "
          f"({os.path.getsize(zip_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
