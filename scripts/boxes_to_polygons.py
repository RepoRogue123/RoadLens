"""
Turn pothole BOXES into pothole OUTLINES with SAM 2, so box datasets can train the segmenter.

RDD2022 and the Mendeley set mark each pothole with a rectangle. Training a segmenter on
rectangles teaches it that potholes are rectangles, which is why segmenter v2 left both
out. SAM 2, prompted with the rectangle, returns the object inside it.

That is only worth doing if the returned outline is a good one, and on RDD there is no true
outline to check against. So the check is made where truth exists:

  --gate      On the held-out PothRGBD and Pothole-600 test photos, draw the box around
              each TRUE outline, enlarge it by 10% (hand-drawn boxes are loose), prompt
              SAM 2, and compare its mask with the true outline. Two prompts are tried
              (box alone; box plus a click at its centre) and the better median IoU is
              the one --convert uses. Rule, fixed beforehand: convert only if the median
              IoU is at least 0.70 on both sets.

  --convert   For every frame the manifest marks box_train / box_test
              (scripts/build_corpus_manifest.py): one mask per box. A mask is rejected if
              it fills more than 90% of its box (SAM returned the rectangle), less than
              15% (it found something small inside), or is under 100 px at the 640 scale.
              A frame is used only if EVERY box in it got an accepted mask: a frame with
              one pothole outlined and another left out would teach that the second is road.

  --sheet     Contact sheet of random conversions, for looking at.

Output: archive/corpus_v3/outlines/<name>.txt (YOLO polygons), conversion_log.csv,
        gate.json, sheet_<source>.jpg

Usage:
    python scripts/boxes_to_polygons.py --gate
    python scripts/boxes_to_polygons.py --convert [--limit N]
    python scripts/boxes_to_polygons.py --sheet
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from mask_refine import masks_from_boxes                                 # noqa: E402
from scripts.eval_yolo_seg import gt_masks, iou                          # noqa: E402

OUT = os.path.join(PROJECT_DIR, "archive", "corpus_v3")
V2 = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
OUTLINES = os.path.join(OUT, "outlines")
GATE = os.path.join(OUT, "gate.json")
MIN_GATE_IOU = 0.70
MAX_FILL, MIN_FILL, MIN_PX_AT_640 = 0.90, 0.15, 100
LOOSEN = 0.10
MAX_SIDE = 1280                      # Norway's frames are 3650 px wide; SAM 2 works at 1024
SEED = 20261003


def loosen(box, w, h, by=LOOSEN):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * by / 2, (y1 - y0) * by / 2
    return [max(0, x0 - dx), max(0, y0 - dy), min(w - 1, x1 + dx), min(h - 1, y1 + dy)]


def gate():
    rows = []
    for split in ("test_pothrgbd", "test_p600"):
        for p in sorted(glob.glob(os.path.join(V2, split, "images", "*"))):
            bgr = cv2.imread(p)
            h, w = bgr.shape[:2]
            truth = gt_masks(os.path.join(V2, split, "labels", os.path.splitext(os.path.basename(p))[0] + ".txt"), h, w)
            truth = [m for m in truth if int(m.sum()) >= 100]
            if not truth:
                continue
            boxes = []
            for m in truth:
                ys, xs = np.nonzero(m)
                boxes.append(loosen([xs.min(), ys.min(), xs.max(), ys.max()], w, h))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            for prompt, centre in (("box", False), ("box+centre", True)):
                for m, b, pm in zip(truth, boxes, masks_from_boxes(rgb, boxes, centre_point=centre)):
                    fill = float(pm.sum()) / max((b[2] - b[0]) * (b[3] - b[1]), 1) if pm is not None else 0.0
                    rows.append({"set": split, "prompt": prompt, "iou": iou(pm, m) if pm is not None else 0.0, "fill": fill,
                                 "accepted": pm is not None and MIN_FILL <= fill <= MAX_FILL})
        print(f"  {split} done", flush=True)
    d = pd.DataFrame(rows)
    res = {}
    print(f"\n{'set':14s} {'prompt':11s} {'n':>4s} {'median IoU':>11s} {'IoU>=0.5':>9s} {'accepted':>9s} {'median IoU of accepted':>23s}")
    for (s, pr), g in d.groupby(["set", "prompt"]):
        a = g[g.accepted]
        res[f"{s}|{pr}"] = {"n": int(len(g)), "median_iou": float(g.iou.median()), "share_iou_0.5": float((g.iou >= 0.5).mean()),
                           "accepted": float(g.accepted.mean()), "median_iou_accepted": float(a.iou.median()) if len(a) else None}
        r = res[f"{s}|{pr}"]
        print(f"{s:14s} {pr:11s} {r['n']:4d} {r['median_iou']:11.3f} {r['share_iou_0.5']:9.1%} {r['accepted']:9.1%} {r['median_iou_accepted'] or 0:23.3f}")
    worst = {pr: min(res[f"{s}|{pr}"]["median_iou_accepted"] or 0 for s in ("test_pothrgbd", "test_p600")) for pr in ("box", "box+centre")}
    best = max(worst, key=worst.get)
    passed = worst[best] >= MIN_GATE_IOU
    res.update({"chosen_prompt": best, "worst_set_median_iou": worst[best], "passed": bool(passed), "rule": f"median IoU of accepted masks >= {MIN_GATE_IOU} on both sets"})
    os.makedirs(OUT, exist_ok=True)
    with open(GATE, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nprompt chosen: {best} (worst-set median IoU of accepted masks {worst[best]:.3f}); gate {'PASSED' if passed else 'FAILED'} at {MIN_GATE_IOU}")


def polygon_lines(mask):
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)                       # one outline per box: the largest piece
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True)
    pts = c.reshape(-1, 2).astype(np.float64) / [w, h]
    return ["0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in np.clip(pts, 0, 1))] if len(pts) >= 3 else []


def convert(limit):
    with open(GATE, encoding="utf-8") as f:
        g = json.load(f)
    if not g["passed"]:
        sys.exit("the gate did not pass; no conversion (see archive/corpus_v3/gate.json)")
    centre = g["chosen_prompt"] == "box+centre"
    m = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    m = m[m.role.isin(["box_train", "box_test"])]
    if limit:
        m = m.groupby(["source", "country"], group_keys=False).head(limit)
    os.makedirs(OUTLINES, exist_ok=True)
    log = []
    for i, r in enumerate(m.itertuples(), start=1):
        bgr = cv2.imread(os.path.join(PROJECT_DIR, r.path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        s = min(1.0, MAX_SIDE / max(h, w))
        if s < 1:
            bgr = cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]
        boxes = [[v * s for v in b] for b in json.loads(r.boxes)]
        masks = masks_from_boxes(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), boxes, centre_point=centre)
        lines, ok = [], True
        for b, pm in zip(boxes, masks):
            area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
            fill = float(pm.sum()) / area if pm is not None else 0.0
            px640 = float(pm.sum()) * (640 / max(h, w)) ** 2 if pm is not None else 0.0
            reason = ("failed" if pm is None else "rectangle" if fill > MAX_FILL else "too_small_in_box" if fill < MIN_FILL
                      else "tiny" if px640 < MIN_PX_AT_640 else "")
            pl = polygon_lines(pm) if not reason else []
            if not pl and not reason:
                reason = "no_polygon"
            ok = ok and not reason
            lines += pl
            log.append({"name": r.name, "source": r.source, "country": r.country, "role": r.role, "fill": round(fill, 3), "rejected": reason})
        if ok and lines:
            with open(os.path.join(OUTLINES, r.name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        if i % 250 == 0:
            print(f"  {i}/{len(m)}", flush=True)
    d = pd.DataFrame(log)
    d.to_csv(os.path.join(OUT, "conversion_log.csv"), index=False)
    frames = d.groupby(["source", "country", "name"]).rejected.apply(lambda v: (v == "").all()).reset_index(name="usable")
    print(f"\nboxes: {len(d)}; accepted {int((d.rejected == '').sum())} ({(d.rejected == '').mean():.1%}); rejected by reason: {d[d.rejected != ''].rejected.value_counts().to_dict()}")
    print("usable frames (every box accepted):")
    print(frames.groupby(["source", "country"]).usable.agg(["sum", "count"]).rename(columns={"sum": "usable", "count": "frames"}).to_string())


def sheet(n=60):
    rng = np.random.default_rng(SEED)
    m = pd.read_csv(os.path.join(OUT, "manifest.csv"))
    m = m[m.role.isin(["box_train", "box_test"])]
    have = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(OUTLINES, "*.txt"))}
    m = m[m.name.isin(have)]
    for group, g in m.groupby(m.source.where(m.source != "rdd", "rdd_" + m.country)):
        tiles = []
        for r in g.loc[rng.choice(g.index, min(n // 6 * 2, len(g)), replace=False)].itertuples():
            bgr = cv2.imread(os.path.join(PROJECT_DIR, r.path))
            h, w = bgr.shape[:2]
            for b in json.loads(r.boxes):
                cv2.rectangle(bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), max(1, w // 400))
            for mk in gt_masks(os.path.join(OUTLINES, r.name + ".txt"), h, w):
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(bgr, cs, -1, (0, 0, 255), max(2, w // 300))
            tiles.append(cv2.resize(bgr, (320, 240)))
        while len(tiles) % 5:
            tiles.append(np.zeros((240, 320, 3), np.uint8))
        img = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        out = os.path.join(OUT, f"sheet_{group}.jpg")
        cv2.imwrite(out, img)
        print(f"wrote {os.path.relpath(out, PROJECT_DIR)}  (yellow: the box; red: SAM 2's outline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N frames per country (a trial)")
    args = ap.parse_args()
    if args.gate:
        gate()
    if args.convert:
        convert(args.limit)
    if args.sheet:
        sheet()


if __name__ == "__main__":
    main()
