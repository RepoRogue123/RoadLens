"""
The other common published approach: classify severity straight from the pothole picture.

Many pothole-severity papers train an image classifier to output Shallow / Moderate / Deep.
To compare fairly, this one gets everything ours gets: the same measured labels, the same
capture-session split, the same human outlines. It sees a square crop around each pothole
(1.5x its box), and is an ImageNet-pretrained ResNet-18 fine-tuned with class-balanced
cross-entropy, flips and colour jitter, epoch chosen on the validation sessions by the safety-
weighted cost (under-reporting counts 3x over-reporting) that our model is judged by.

Reports, on the 208 test-session potholes: correct, safety-weighted cost per pothole,
under-reported, Deep found. Writes ml_results/comparison/severity_classifier.json.

Usage:
    python scripts/train_severity_classifier.py [--epochs 25]
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.pothrgbd_metric_labels import DATA_DIR, load_polygons, timestamp_key   # noqa: E402
from scripts.train_metric_regressor import BAND_INDEX, policy_cost                   # noqa: E402
from scripts.train_relief_depth import split_keys                                    # noqa: E402

OUT = os.path.join(PROJECT_DIR, "ml_results", "comparison")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SEED = 20261007


def crops(keys):
    t = pd.read_csv(os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv"))
    t = t[(t.mask_source == "gt") & t.key.isin(set(keys))].drop_duplicates(["key", "pothole"])
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    out = []
    for key, g in t.groupby("key"):
        bgr = cv2.imread(imgs[key])
        h, w = bgr.shape[:2]
        polys = load_polygons(labs[key], h, w)
        for r in g.itertuples():
            ys, xs = np.nonzero(polys[int(r.pothole) - 1])
            cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
            s = max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.75 + 8
            y0, y1, x0, x1 = int(max(0, cy - s)), int(min(h, cy + s)), int(max(0, cx - s)), int(min(w, cx + s))
            crop = cv2.resize(bgr[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA)
            out.append((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), BAND_INDEX[r.gt_severity]))
    return out


def tensor(batch, rng, train):
    xs = []
    for img, _ in batch:
        x = img.astype(np.float32) / 255.0
        if train:
            if rng.random() < 0.5:
                x = x[:, ::-1]
            x = np.clip(x * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08), 0, 1)
        xs.append(((x - MEAN) / STD).transpose(2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(np.stack(xs))), torch.tensor([y for _, y in batch])


def evaluate(model, data, dev):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            x, _ = tensor(data[i:i + 64], None, False)
            preds.append(model(x.to(dev)).argmax(1).cpu().numpy())
    yhat, y = np.concatenate(preds), np.array([v for _, v in data])
    return {"n": int(len(y)), "correct": float((yhat == y).mean()), "cost": policy_cost(y, yhat) / len(y),
            "under_reported": int((yhat < y).sum()), "deep_found": int(((yhat == 2) & (y == 2)).sum()),
            "n_deep": int((y == 2).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    args = ap.parse_args()
    import torchvision
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = (crops(split_keys(s)) for s in ("train", "val", "test_pothrgbd"))
    counts = np.bincount([y for _, y in train], minlength=3)
    print(f"crops: train {len(train)} {counts.tolist()}, val {len(val)}, test {len(test)}", flush=True)

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (3 * np.maximum(counts, 1)), dtype=torch.float32).to(dev))
    best, best_state = None, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for i in range(0, len(order), 32):
            x, y = tensor([train[j] for j in order[i:i + 32]], rng, True)
            loss = loss_fn(model(x.to(dev)), y.to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = evaluate(model, val, dev)
        if best is None or v["cost"] < best["cost"]:
            best, best_state = dict(v, epoch=epoch), {k: t.detach().clone() for k, t in model.state_dict().items()}
        print(f"epoch {epoch:2d}  val cost {v['cost']:.3f}  correct {v['correct']:.1%}", flush=True)
    model.load_state_dict(best_state)
    res = {"val": best, "test": evaluate(model, test, dev)}
    t = res["test"]
    print(f"\ntest ({t['n']} potholes): correct {t['correct']:.1%}  cost {t['cost']:.3f}  under-reported {t['under_reported']}  "
          f"Deep found {t['deep_found']}/{t['n_deep']}  (epoch {best['epoch']} chosen on validation)")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "severity_classifier.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
