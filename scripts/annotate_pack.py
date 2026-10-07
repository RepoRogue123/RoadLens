"""
Label a RoadLens annotation pack. Needs only Python 3.8+, opencv-python and numpy.

A pack is a folder built by scripts/build_annotation_pack.py:

    manifest.csv   one row per pothole: item_id, frozen outline (polygon), who labels it
    task.json      the question, the keys, the pack version
    items/*.jpg    what you see: the photo with the pothole outlined, and a zoomed crop

You see only the potholes assigned to you. Your answers go to
labels_<task>_v<version>_<your name>.csv in the pack folder — send that one file back.
Label on your own: do not look at or discuss anyone else's answers until the merge.

Usage (inside the unzipped pack folder):
    python annotate_pack.py --annotator vyankatesh
    python annotate_pack.py --annotator vyankatesh --dry-run      # list your queue, no window
    python annotate_pack.py --annotator consensus --review disputed.csv   # joint review call

Keys: shown on screen (from task.json), plus  s = skip   u = undo   q = save and quit.
Your progress is saved after every key press; run the same command again to continue.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_FIELDS = ["item_id", "label", "annotator", "labelled_at", "pack_version"]


# ── pack format helpers (also imported by the coordinator's scripts) ─────────
def polygon_from_mask(mask):
    """Largest outer contour of a binary mask as normalised [[x, y], ...] (JSON-safe)."""
    h, w = mask.shape[:2]
    cnts, _ = cv2.findContours((np.asarray(mask) > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    return [[round(float(x) / w, 6), round(float(y) / h, 6)] for x, y in c]


def mask_from_polygon(polygon, h, w):
    """Rasterise a normalised polygon (list or JSON string) back to an HxW uint8 mask."""
    if isinstance(polygon, str):
        polygon = json.loads(polygon)
    pts = np.round(np.array(polygon, dtype=np.float64) * [w, h]).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load_pack(pack_dir):
    with open(os.path.join(pack_dir, "task.json"), encoding="utf-8") as f:
        task = json.load(f)
    return task, read_csv(os.path.join(pack_dir, "manifest.csv"))


def labels_path(pack_dir, task, annotator):
    return os.path.join(pack_dir, f"labels_{task['task']}_v{task['version']}_{annotator}.csv")


def queue_for(manifest, annotator, review_ids=None):
    """Items this annotator must label, in manifest order."""
    if review_ids is not None:
        return [r for r in manifest if r["item_id"] in review_ids]
    return [r for r in manifest if annotator in r["assigned_to"].split(";")]


# ── labelling session ─────────────────────────────────────────────────────────
def draw(view, caption, legend):
    bar = np.zeros((64, view.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, caption, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(bar, legend, (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 235, 235), 1)
    return np.vstack([view, bar])


class OpenCVWindow:
    """The usual window. Needs an OpenCV build with GUI support (opencv-python)."""

    def __init__(self, title):
        self.title = title
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    def show(self, image_bgr):
        cv2.imshow(self.title, image_bgr)

    def key(self):
        return cv2.waitKey(0) & 0xFF

    def close(self):
        cv2.destroyAllWindows()


class TkWindow:
    """
    Fallback when OpenCV has no GUI (opencv-python-headless, which other packages
    often install silently). tkinter ships with Python on Windows and macOS; images
    go through PNG so no extra imaging library is needed. Closing the window = quit.
    """

    def __init__(self, title):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.pressed = tk.IntVar(value=0)
        self.root.bind("<Key>", lambda e: self.pressed.set(ord(e.char.lower()) if e.char else 0))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.pressed.set(ord("q")))
        self.max_w = int(self.root.winfo_screenwidth() * 0.9)
        self.max_h = int(self.root.winfo_screenheight() * 0.85)

    def show(self, image_bgr):
        import base64
        h, w = image_bgr.shape[:2]
        s = min(1.0, self.max_w / w, self.max_h / h)
        if s < 1.0:
            image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", image_bgr)
        self.photo = self.tk.PhotoImage(data=base64.b64encode(png.tobytes()))
        self.label.configure(image=self.photo)
        self.root.lift()
        self.root.focus_force()

    def key(self):
        self.pressed.set(0)
        while self.pressed.get() == 0:
            self.root.wait_variable(self.pressed)
        return self.pressed.get()

    def close(self):
        self.root.destroy()


def open_window(title):
    """OpenCV's window if this OpenCV build can show one, otherwise tkinter."""
    try:
        return OpenCVWindow(title)
    except cv2.error:
        try:
            print("  (this OpenCV has no window support - using the built-in tkinter window instead)")
            return TkWindow(title)
        except Exception as e:
            sys.exit("Cannot open a window: OpenCV has no GUI support and tkinter is unavailable "
                     f"({e}).\nFix: pip uninstall -y opencv-python-headless && pip install opencv-python")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", required=True, help="your first name, lower case, as listed in task.json")
    ap.add_argument("--pack", default=HERE, help="pack folder (default: the folder this script is in)")
    ap.add_argument("--review", default=None, help="disputed.csv from the merge, for the joint review")
    ap.add_argument("--dry-run", action="store_true", help="print your queue and exit")
    args = ap.parse_args()

    task, manifest = load_pack(args.pack)
    name = args.annotator.strip().lower()
    review_ids = None
    if args.review:
        review_ids = {r["item_id"] for r in read_csv(args.review)}
    elif name not in task["annotators"]:
        sys.exit(f"Unknown annotator '{name}'. Use one of: {', '.join(task['annotators'])}")

    queue = queue_for(manifest, name, review_ids)
    out = labels_path(args.pack, task, name)
    done = read_csv(out)
    done_ids = {r["item_id"] for r in done}
    todo = [r for r in queue if r["item_id"] not in done_ids]
    print(f"{task['title']}  (pack v{task['version']})")
    print(f"  {name}: {len(queue)} potholes assigned, {len(done_ids & {r['item_id'] for r in queue})} done, "
          f"{len(todo)} to go  ->  {os.path.basename(out)}")
    if args.dry_run:
        for r in todo[:10]:
            print(f"    {r['item_id']}  {r['source']}")
        return

    keys = {ord(k): v for k, v in task["keys"].items()}
    legend = "   ".join(f"{k}={v}" for k, v in task["keys"].items()) + "   s=skip  u=undo  q=quit"
    window = open_window(f"RoadLens - {task['question']}")
    i = 0
    while i < len(todo):
        r = todo[i]
        view = cv2.imread(os.path.join(args.pack, "items", f"{r['item_id']}.jpg"))
        if view is None:
            print(f"  ! missing image for {r['item_id']}, skipped")
            i += 1
            continue
        n_done = len(done_ids & {q["item_id"] for q in queue})
        window.show(draw(view, f"[{n_done + 1}/{len(queue)}]  {task['question']}", legend))
        k = window.key()
        if k == ord("q"):
            break
        if k == ord("s"):
            i += 1
            continue
        if k == ord("u"):
            if done:
                removed = done.pop()
                done_ids.discard(removed["item_id"])
                write_csv(out, done, LABEL_FIELDS)
                back = next((j for j, t in enumerate(todo) if t["item_id"] == removed["item_id"]), None)
                if back is None:
                    todo.insert(i, next(q for q in queue if q["item_id"] == removed["item_id"]))
                else:
                    i = back
                print(f"  undid {removed['item_id']} ({removed['label']})")
            continue
        if k in keys:
            done.append({"item_id": r["item_id"], "label": keys[k], "annotator": name,
                         "labelled_at": datetime.now().isoformat(timespec="seconds"),
                         "pack_version": task["version"]})
            done_ids.add(r["item_id"])
            write_csv(out, done, LABEL_FIELDS)        # saved after every answer
            i += 1
    window.close()
    counts = {}
    for d in done:
        counts[d["label"]] = counts.get(d["label"], 0) + 1
    left = len([q for q in queue if q["item_id"] not in done_ids])
    print(f"saved {len(done)} answers to {os.path.basename(out)}: {counts}")
    print("  all done - send this file back." if not left else f"  {left} still to do - run the same command to continue.")


if __name__ == "__main__":
    main()
