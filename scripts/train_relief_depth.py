"""
Fine-tune Depth-Anything-V2 (ViT-S) to predict relief below the road, on PothRGBD.

Target, per frame: fit one robust plane to the RealSense depth of the road (valid
pixels outside the dilated pothole polygons, iteratively trimmed), then
relief = depth - plane, in mm, positive = deeper. Pixels with no reading, or more than
300 mm off the plane (kerbs, objects), carry no loss. Frames whose road is not planar
(residual std > 25 mm) are left out of training.

Split: the segmenter's capture-session split (archive/yolo_seg_v2), so the sessions held
out here are held out from the segmenter as well. Train on its train sessions, pick the
epoch on its val sessions by POTHOLE-level error (the label protocol applied to the
prediction, human masks), and never touch test_pothrgbd.

Usage:
    python scripts/train_relief_depth.py [--epochs 40]
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import relief_depth as R                                                        # noqa: E402
from scripts.pothrgbd_metric_labels import (                                    # noqa: E402
    DATA_DIR, DEPTH_VALID_MIN, load_polygons, timestamp_key,
)

SPLIT_DIR = os.path.join(PROJECT_DIR, "archive", "yolo_seg_v2")
RUN_DIR = os.path.join(PROJECT_DIR, "archive", "relief_runs", "vits")
TRAINSET = os.path.join(PROJECT_DIR, "ml_results", "pothrgbd", "trainset_metric.csv")
MAX_ROAD_STD_MM = 25.0
MAX_ABS_RELIEF_MM = 300.0
POTHOLE_WEIGHT = 3.0
SEED = 20261002
STRONG_AUG = True                # set by --no-strong-aug
ZOOM_MIN = 0.7                   # set by --zoom-min
EMA_DECAY = 0.995
P600_TRAIN = os.path.join(PROJECT_DIR, "pothole600", "training")
P600_SIDE = 392                  # Pothole-600 frames are square; 28 patches of 14 px
P600_BATCH = 4
STYLE_BANK = os.path.join(PROJECT_DIR, "archive", "style_bank", "amp.npy")
CROPS_DIR = os.path.join(PROJECT_DIR, "archive", "corpus_v3", "unlabelled_crops")
CONSISTENCY_BATCH = 4
RSRD_CACHE = os.path.join(PROJECT_DIR, "archive", "relief_targets", "rsrd")
RSRD_BATCH = 4
CONSISTENCY_RAMP_EPOCHS = 5      # the teacher is useless until it has learned something


def split_keys(split):
    return sorted({os.path.basename(p)[len("pothrgbd_"):].split(".")[0]
                   for p in glob.glob(os.path.join(SPLIT_DIR, split, "images", "pothrgbd_*"))})


def relief_target(depth_mm, polys, exact=False):
    """
    (relief_mm, valid, pothole_mask, road_std) for one frame.

    `exact` fits the road plane to INVERSE depth. A flat road seen by a pinhole camera is
    exactly linear in 1/depth across the image and only approximately linear in depth; the
    approximation holds for a camera looking nearly straight down from a distance and
    breaks for a tilted camera close to the road (the Fan stereo frames: 25 mm of false
    relief across the frame from the linear fit).
    """
    h, w = depth_mm.shape
    d = depth_mm.astype(np.float64)
    ok = d >= DEPTH_VALID_MIN
    pot = np.zeros((h, w), np.uint8)
    for m in polys:
        pot |= m
    road = ok & (cv2.dilate(pot, np.ones((15, 15), np.uint8)) == 0)
    ys, xs = np.nonzero(road)
    if len(ys) < 2000:
        return None
    zs = d[ys, xs]
    fit = 1.0 / zs if exact else zs
    plane = (lambda c, x, y: 1.0 / (c[0] * x + c[1] * y + c[2])) if exact else (lambda c, x, y: c[0] * x + c[1] * y + c[2])
    keep = np.ones(len(zs), bool)
    for _ in range(3):
        a = np.column_stack([xs[keep], ys[keep], np.ones(int(keep.sum()))])
        c, *_ = np.linalg.lstsq(a, fit[keep], rcond=None)
        r = zs - plane(c, xs, ys)
        keep = np.abs(r) < 2.5 * r[keep].std()
    gy, gx = np.mgrid[0:h, 0:w]
    with np.errstate(divide="ignore", invalid="ignore"):
        relief = d - plane(c, gx, gy)
    valid = ok & np.isfinite(relief) & (np.abs(relief) < MAX_ABS_RELIEF_MM)
    return np.nan_to_num(relief, posinf=0.0, neginf=0.0).astype(np.float32), valid, pot, float(r[keep].std())


ABSOLUTE_TARGET = False          # set by --target absolute


def load_frames(keys, exact=False):
    imgs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "images", "*.jpg"))}
    deps = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "depths", "*.npy"))}
    labs = {timestamp_key(p): p for p in glob.glob(os.path.join(DATA_DIR, "labels", "*.txt"))}
    out = []
    for k in keys:
        if k not in imgs or k not in deps or k not in labs:
            continue
        depth = np.load(deps[k])
        h, w = depth.shape
        bgr = cv2.imread(imgs[k])
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        polys = load_polygons(labs[k], h, w)
        t = relief_target(depth, polys, exact=exact)
        if t is None:
            continue
        if ABSOLUTE_TARGET:
            # The comparison with published work (PLOS ONE 2026, SHM 2023): train on plain depth
            # from the camera, as they do, and read the pothole with the same ring-of-road
            # protocol afterwards. Everything else in the recipe is unchanged.
            ok = depth >= DEPTH_VALID_MIN
            t = (np.where(ok, depth, 0).astype(np.float32), ok, t[2], t[3])
        out.append({"key": k, "bgr": bgr, "polys": polys, "relief": t[0], "valid": t[1], "pot": t[2], "road_std": t[3]})
    return out


def apply_style(bgr, bank, rng):
    """
    Re-light a frame like a photo from another dataset (Fourier domain adaptation).

    Replaces the lowest frequencies of the frame's amplitude spectrum with those of a random
    photo in the style bank (scripts/build_style_bank.py). Phase is untouched, so every edge
    stays put and the depth target remains valid.

    Kept deliberately small. Four times in five only the zero frequency is swapped: the
    other photo's colour cast and exposure. Otherwise the 3x3 block is blended in at 30%,
    a gentle illumination gradient. Larger blocks were tried and rejected on sight: a road
    close-up has almost no low-frequency energy and a dashcam scene has a great deal (sky
    over ground), so the swap paints coloured bands across the road that no camera produces
    and that read as shading.
    """
    block = bank[int(rng.integers(len(bank)))]
    half = block.shape[0] // 2
    b, mix = (0, 1.0) if rng.random() < 0.8 else (1, 0.3)
    f = np.fft.fft2(bgr.astype(np.float32), axes=(0, 1))
    amp, phase = np.fft.fftshift(np.abs(f), axes=(0, 1)), np.angle(f)
    ch, cw = bgr.shape[0] // 2, bgr.shape[1] // 2
    win = (slice(ch - b, ch + b + 1), slice(cw - b, cw + b + 1))
    amp[win] = (1 - mix) * amp[win] + mix * block[half - b:half + b + 1, half - b:half + b + 1]
    out = np.fft.ifft2(np.fft.ifftshift(amp, axes=(0, 1)) * np.exp(1j * phase), axes=(0, 1)).real
    return np.clip(out, 0, 255).astype(np.uint8)


def load_shape_frames():
    """Pothole-600 training photos with their depth SHAPE (larger = deeper; no millimetres)."""
    out = []
    for p in sorted(glob.glob(os.path.join(P600_TRAIN, "rgb", "*.png"))):
        name = os.path.basename(p)
        index, _ = R.jet_to_scalar(cv2.imread(os.path.join(P600_TRAIN, "tdisp", name)))
        lab = cv2.imread(os.path.join(P600_TRAIN, "label", name), cv2.IMREAD_GRAYSCALE)
        out.append({"bgr": cv2.imread(p), "shape": 255.0 - index.astype(np.float32), "pot": (lab > 127).astype(np.uint8)})
    return out


def shape_batch(frames, idx, rng, style_bank=None, style_p=0.0):
    """Square 392x392 batch from Pothole-600: image, shape target, weights (0 outside the frame)."""
    xs, ys, ws = [], [], []
    size = (P600_SIDE, P600_SIDE)
    for i in idx:
        f = frames[i]
        bgr, shape, pot = f["bgr"], f["shape"], f["pot"]
        valid = np.ones(pot.shape, np.uint8)
        h, w = pot.shape
        m = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-0.1, 0.1) * w, h / 2 + rng.uniform(-0.1, 0.1) * h),
                                    rng.uniform(-15, 15), rng.uniform(0.7, 1.4))
        bgr = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        shape, valid, pot = (cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_NEAREST) for a in (shape, valid, pot))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        if style_bank is not None and rng.random() < style_p:
            bgr = apply_style(cv2.resize(bgr, (R.IN_W, R.IN_H)), style_bank, rng)
        bgr = np.clip(bgr.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        x = R.to_tensor(bgr, size)
        y = cv2.resize(shape, size, interpolation=cv2.INTER_NEAREST)
        wgt = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST).astype(np.float32) \
            * (1.0 + (POTHOLE_WEIGHT - 1.0) * cv2.resize(pot, size, interpolation=cv2.INTER_NEAREST))
        if rng.random() < 0.5:
            x, y, wgt = x[:, :, ::-1], y[:, ::-1], wgt[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
        ws.append(np.ascontiguousarray(wgt))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws)))


def shape_loss(out, y, w):
    """
    Scale-and-shift-invariant L1 (as MiDaS): both maps are centred on their median and
    divided by their mean absolute deviation, per image, before comparing. Only the SHAPE
    of the prediction is supervised; how many millimetres it spans is left to PothRGBD.
    """
    total = 0.0
    for o, t, wt in zip(out, y, w):
        ok = wt > 0
        if int(ok.sum()) < 100:
            continue

        def norm(v):
            v = v - v[ok].median()
            return v / v[ok].abs().mean().clamp(min=0.02)
        total = total + ((norm(o) - norm(t)).abs() * wt).sum() / wt.sum()
    return total / len(out)


def consistency_batch(files, rng, style_bank=None):
    """
    Two views of unlabelled pothole close-ups (scripts/build_unlabelled_crops.py):
    the photo as it is, and the photo mirrored, re-lit and brightness-shifted.
    """
    plain, changed = [], []
    for p in rng.choice(files, CONSISTENCY_BATCH, replace=False):
        bgr = cv2.imread(os.path.join(CROPS_DIR, p))
        plain.append(R.to_tensor(bgr))
        alt = apply_style(bgr, style_bank, rng) if style_bank is not None else bgr
        alt = np.clip(alt.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        changed.append(np.ascontiguousarray(R.to_tensor(alt)[:, :, ::-1]))
    return torch.from_numpy(np.stack(plain)), torch.from_numpy(np.stack(changed))


def rsrd_keys(role):
    """RSRD frames cached by scripts/build_relief_targets.py, for one drive-level split role."""
    from scripts.depth_sources import rsrd_split
    keys = sorted(f[:-4] for f in os.listdir(RSRD_CACHE) if f.endswith(".npz"))
    split = rsrd_split(keys)
    return [k for k in keys if split[k] == role]


def rsrd_batch(keys, rng, style_bank=None, style_p=0.0):
    """
    A batch of RSRD frames: vehicle camera, road 2-9 m ahead, fused-LiDAR relief.

    Frames are 16:9; a random 4:3 window is cut so nothing is stretched. Only ~16% of pixels
    carry a reading, and only they carry loss; measured depressions (> 15 mm) weigh 3x, as
    pothole pixels do in PothRGBD. Frames whose road is far from planar are skipped.
    """
    xs, ys, ws = [], [], []
    size = (R.IN_W, R.IN_H)
    while len(xs) < RSRD_BATCH:
        z = np.load(os.path.join(RSRD_CACHE, str(rng.choice(keys)) + ".npz"))
        if float(z["road_std"]) > MAX_ROAD_STD_MM:
            continue
        bgr, relief, valid, pot = z["bgr"], z["relief"], z["valid"].astype(np.uint8), z["mask"]
        h, w = relief.shape
        cw = min(w, int(round(h * R.IN_W / R.IN_H)))
        a = int(rng.integers(0, w - cw + 1))
        bgr, relief, valid, pot = (v[:, a:a + cw] for v in (bgr, relief, valid, pot))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
        if style_bank is not None and rng.random() < style_p:
            bgr = apply_style(bgr, style_bank, rng)
        bgr = np.clip(bgr.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
        x = R.to_tensor(bgr)
        y = cv2.resize(relief, size, interpolation=cv2.INTER_NEAREST) / R.RELIEF_UNIT_MM
        wgt = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST).astype(np.float32) \
            * (1.0 + (POTHOLE_WEIGHT - 1.0) * cv2.resize(pot, size, interpolation=cv2.INTER_NEAREST))
        if rng.random() < 0.5:
            x, y, wgt = x[:, :, ::-1], y[:, ::-1], wgt[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
        ws.append(np.ascontiguousarray(wgt))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws)))


def batch_of(frames, idx, train, rng, style_bank=None, style_p=0.0):
    xs, ys, ws = [], [], []
    size = (R.IN_W, R.IN_H)
    for i in idx:
        f = frames[i]
        bgr, relief, valid, pot = f["bgr"], f["relief"], f["valid"].astype(np.uint8), f["pot"]
        if train and style_bank is not None and rng.random() < style_p:
            bgr = apply_style(cv2.resize(bgr, size, interpolation=cv2.INTER_AREA), style_bank, rng)
            bgr = cv2.resize(bgr, (relief.shape[1], relief.shape[0]), interpolation=cv2.INTER_LINEAR)
        if train:
            # brightness/contrast jitter: lighting varies far more in the wild than in PothRGBD
            bgr = np.clip(bgr.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
            if STRONG_AUG:
                # Zoom, rotate, shift. Relief in mm does not change under any of these, and
                # zoom stands in for a camera held closer or further than PothRGBD's.
                h, w = relief.shape
                # ZOOM_MIN below 0.7 (--zoom-min) shrinks the pothole in the frame: in PothRGBD it
                # fills a fifth of the picture, and the network otherwise learns "small = shallow".
                zoom = float(np.exp(rng.uniform(np.log(ZOOM_MIN), np.log(1.4))))
                m = cv2.getRotationMatrix2D((w / 2 + rng.uniform(-0.1, 0.1) * w, h / 2 + rng.uniform(-0.1, 0.1) * h),
                                            rng.uniform(-15, 15), zoom)
                bgr = cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_AREA if zoom < 0.6 else cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)
                relief = cv2.warpAffine(relief, m, (w, h), flags=cv2.INTER_NEAREST)
                valid = cv2.warpAffine(valid, m, (w, h), flags=cv2.INTER_NEAREST)      # 0 outside the frame
                pot = cv2.warpAffine(pot, m, (w, h), flags=cv2.INTER_NEAREST)
        x = R.to_tensor(bgr)
        y = cv2.resize(relief, size, interpolation=cv2.INTER_NEAREST) / R.RELIEF_UNIT_MM
        v = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        p = cv2.resize(pot, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        wgt = v * (1.0 + (POTHOLE_WEIGHT - 1.0) * p)
        if train and rng.random() < 0.5:
            x, y, wgt = x[:, :, ::-1], y[:, ::-1], wgt[:, ::-1]
        xs.append(np.ascontiguousarray(x))
        ys.append(np.ascontiguousarray(y))
        ws.append(np.ascontiguousarray(wgt))
    return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ws)))


def remove_road_plane(err, road):
    """
    Subtract from each error map the plane that best fits it over the road pixels.

    The label is measured from a plane fitted LOCALLY around each pothole, so a smooth
    tilt or offset across the frame never reaches the answer. But the training target is
    measured from one plane per frame, and real frames are not perfectly planar (camber,
    lens-dependent depth bias), so without this the network spends its capacity on, and
    is penalised for, a frame-wide shape that the label ignores.
    """
    b, h, w = err.shape
    ys = torch.linspace(-1, 1, h, device=err.device).view(1, h, 1).expand(b, h, w)
    xs = torch.linspace(-1, 1, w, device=err.device).view(1, 1, w).expand(b, h, w)
    a = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(b, -1, 3)          # B, N, 3
    wt = road.reshape(b, -1, 1)
    m = (a * wt).transpose(1, 2) @ a + 1e-3 * torch.eye(3, device=err.device)
    rhs = (a * wt).transpose(1, 2) @ err.reshape(b, -1, 1)
    coef = torch.linalg.solve(m, rhs)
    return err - (a @ coef).reshape(b, h, w)


def pothole_errors(model, frames, labels):
    """Label protocol on the prediction, human masks. Returns (pred_mm, gt_mm) arrays."""
    pred, gt = [], []
    for f in frames:
        rows = labels.get(f["key"])
        if rows is None:
            continue
        relief = R.predict_relief(f["bgr"], model)
        for j, gt_mm in rows:
            b = R.bowl_depth_mm(relief, f["polys"][j - 1])
            if b is not None:
                pred.append(b)
                gt.append(gt_mm)
    return np.array(pred), np.array(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="vits", help="run folder under archive/relief_runs/")
    ap.add_argument("--no-strong-aug", action="store_true", help="colour jitter + flip only (the v1 recipe)")
    ap.add_argument("--plane-invariant", action="store_true",
                    help="loss ignores any frame-wide plane in the error (see remove_road_plane)")
    ap.add_argument("--exact-plane", action="store_true",
                    help="training target measured from a road plane fitted to inverse depth (see relief_target)")
    ap.add_argument("--style", type=float, default=0.0,
                    help="probability of re-lighting a training frame like a photo from another dataset (see apply_style)")
    ap.add_argument("--p600-shape", type=float, default=0.0,
                    help="weight of the shape-only loss on Pothole-600 training photos (see shape_loss); 0 = off")
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="weight of the consistency loss on unlabelled pothole close-ups from the other datasets; 0 = off")
    ap.add_argument("--rsrd", type=float, default=0.0,
                    help="weight of the loss on RSRD training drives (vehicle camera, LiDAR relief); 0 = off")
    ap.add_argument("--rsrd-batch", type=int, default=4,
                    help="RSRD frames per step. With 8 GB of GPU memory, --batch 6 --rsrd-batch 2 fits; "
                         "8 + 4 spills into system memory and runs ~7x slower")
    ap.add_argument("--resume", action="store_true", help="continue from <run>/last.pth")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop cleanly after this many minutes, at the end of an epoch; continue with --resume")
    ap.add_argument("--zoom-min", type=float, default=0.7,
                    help="smallest zoom in augmentation (0.7 = the v2 recipe; 0.25 makes potholes small in the frame)")
    ap.add_argument("--target", choices=["relief", "absolute"], default="relief",
                    help="relief = mm below the road (ours); absolute = plain depth from the camera, "
                         "as published pothole-depth methods train (for the comparison only)")
    args = ap.parse_args()
    global STRONG_AUG, ZOOM_MIN, RSRD_BATCH, ABSOLUTE_TARGET
    RSRD_BATCH = args.rsrd_batch
    ABSOLUTE_TARGET = args.target == "absolute"
    STRONG_AUG = not args.no_strong_aug
    ZOOM_MIN = args.zoom_min
    run_dir = os.path.join(os.path.dirname(RUN_DIR), args.name)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda"

    t0 = time.time()
    train = [f for f in load_frames(split_keys("train"), args.exact_plane) if f["road_std"] <= MAX_ROAD_STD_MM]
    val = load_frames(split_keys("val"), args.exact_plane)
    d = pd.read_csv(TRAINSET)
    d = d[d.mask_source == "gt"]
    labels = {k: list(zip(g.pothole, g.gt_mm)) for k, g in d.groupby("key")}
    style_bank = np.load(STYLE_BANK) if args.style > 0 else None
    shape_frames = load_shape_frames() if args.p600_shape > 0 else []
    crops = sorted(os.listdir(CROPS_DIR)) if args.consistency > 0 else []
    rsrd = rsrd_keys("train") if args.rsrd > 0 else []
    if rsrd:
        print(f"RSRD: {len(rsrd)} frames from the training drives, weight {args.rsrd}", flush=True)
    if crops:
        print(f"consistency loss on {len(crops)} unlabelled close-ups, weight {args.consistency}", flush=True)
    print(f"train {len(train)} frames, val {len(val)} frames  (loaded in {time.time() - t0:.0f} s)"
          + (f"; style bank {len(style_bank)} photos, p={args.style}" if style_bank is not None else "")
          + (f"; Pothole-600 shape loss on {len(shape_frames)} photos, weight {args.p600_shape}" if shape_frames else ""), flush=True)

    model = R.build_model().to(device)
    enc = list(model.pretrained.parameters())
    head = list(model.depth_head.parameters())
    opt = torch.optim.AdamW([{"params": enc, "lr": 1e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-5, 1e-4], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    # Averaged weights: steadier than any single step on 600 frames, and it makes the
    # choice of epoch matter less.
    import copy
    ema = copy.deepcopy(model).eval()
    for q in ema.parameters():
        q.requires_grad_(False)

    os.makedirs(run_dir, exist_ok=True)
    best, log, first = (np.inf, -1), [], 1
    last = os.path.join(run_dir, "last.pth")
    if args.resume and os.path.isfile(last):
        s = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(s["model"])
        ema.load_state_dict(s["ema"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        scaler.load_state_dict(s["scaler"])
        rng.bit_generator.state = s["rng"]
        best, log, first = tuple(s["best"]), s["log"], s["epoch"] + 1
        print(f"resumed after epoch {s['epoch']} (best so far {best[0]:.2f} mm at epoch {best[1]})", flush=True)
    for epoch in range(first, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for b in range(len(train) // args.batch):
            x, y, w = (t.to(device) for t in batch_of(train, order[b * args.batch:(b + 1) * args.batch], True, rng,
                                                      style_bank, args.style))
            with torch.autocast("cuda"):
                out = R.forward_relief(model, x)
            err = out.float() - y
            if args.plane_invariant:
                err = remove_road_plane(err, (w == 1.0).float())
            # Huber with a 10 mm knee, weighted; computed in float32 for stability
            loss = (F.smooth_l1_loss(err, torch.zeros_like(err), beta=0.2, reduction="none") * w).sum() / w.sum().clamp(min=1)
            if rsrd:
                rx, ry, rw = (t.to(device) for t in rsrd_batch(rsrd, rng, style_bank, args.style))
                with torch.autocast("cuda"):
                    rout = R.forward_relief(model, rx)
                # A road 2-9 m ahead is rarely one plane; the frame-wide tilt is removed from the
                # error as the label protocol removes it around each pothole.
                rerr = remove_road_plane(rout.float() - ry, (rw > 0).float())
                loss = loss + args.rsrd * (F.smooth_l1_loss(rerr, torch.zeros_like(rerr), beta=0.2, reduction="none")
                                           * rw).sum() / rw.sum().clamp(min=1)
            if shape_frames:
                sx, sy, sw = (t.to(device) for t in shape_batch(shape_frames, rng.choice(len(shape_frames), P600_BATCH, replace=False),
                                                                rng, style_bank, args.style))
                with torch.autocast("cuda"):
                    sout = R.forward_relief(model, sx)
                loss = loss + args.p600_shape * shape_loss(sout.float(), sy, sw)
            if crops:
                # Mean teacher: the averaged network reads the plain photo, the trained one
                # reads the mirrored, re-lit photo, and the two must agree. Any frame-wide
                # plane in the difference is removed first: tilt is not what is being asked.
                plain, changed = (t.to(device) for t in consistency_batch(crops, rng, style_bank))
                with torch.no_grad(), torch.autocast("cuda"):
                    target = R.forward_relief(ema, plain).float().flip(-1)
                with torch.autocast("cuda"):
                    cout = R.forward_relief(model, changed)
                diff = remove_road_plane(cout.float() - target, torch.ones_like(target))
                ramp = min(1.0, epoch / CONSISTENCY_RAMP_EPOCHS)
                loss = loss + args.consistency * ramp * F.smooth_l1_loss(diff, torch.zeros_like(diff), beta=0.2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss))
            with torch.no_grad():
                for qe, qm in zip(ema.parameters(), model.parameters()):
                    qe.mul_(EMA_DECAY).add_(qm.detach(), alpha=1 - EMA_DECAY)
        p, g = pothole_errors(ema, val, labels)
        mae = float(np.abs(p - g).mean())
        r = float(np.corrcoef(p, g)[0, 1])
        log.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_pothole_mae_mm": mae, "val_r": r, "n_val": int(len(g))})
        flag = ""
        if mae < best[0]:
            best = (mae, epoch)
            torch.save(ema.state_dict(), os.path.join(run_dir, "best.pth"))
            flag = "  *"
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val pothole MAE {mae:5.2f} mm  r {r:.3f}  (n={len(g)}){flag}", flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                    "epoch": epoch, "best": best, "log": log}, last)
        if args.max_minutes and epoch < args.epochs and (time.time() - t0) / 60 > args.max_minutes:
            print(f"stopping after epoch {epoch} of {args.epochs} (time limit); rerun with --resume", flush=True)
            sys.exit(3)

    with open(os.path.join(run_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump({"best_epoch": best[1], "best_val_mae_mm": best[0], "epochs": log,
                   "train_frames": len(train), "val_frames": len(val),
                   "strong_aug": STRONG_AUG, "ema_decay": EMA_DECAY,
                   "plane_invariant": bool(args.plane_invariant), "exact_plane": bool(args.exact_plane),
                   "style_p": args.style, "p600_shape_weight": args.p600_shape,
                   "consistency_weight": args.consistency, "zoom_min": ZOOM_MIN,
                   "rsrd_weight": args.rsrd, "rsrd_train_frames": len(rsrd), "rsrd_batch": RSRD_BATCH,
                   "batch": args.batch}, f, indent=2)
    print(f"best epoch {best[1]}: val pothole MAE {best[0]:.2f} mm  ({(time.time() - t0) / 60:.1f} min)")
    print(f"  checkpoint: {os.path.relpath(os.path.join(run_dir, 'best.pth'), PROJECT_DIR)}")
    print("  score it, and install it for serving, with scripts/eval_relief_depth.py --weights ... [--install]")


if __name__ == "__main__":
    main()
