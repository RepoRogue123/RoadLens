"""
Readers for measured-depth datasets other than PothRGBD.

Each reader yields one dict per frame, at the dataset's own resolution:

    key        unique frame id, prefixed with the source
    group      what a split must keep together (dataset number, drive, session)
    bgr        HxWx3 uint8
    depth_mm   HxW float32, distance along the camera axis in mm, 0 = no reading
    mask       HxW uint8, pothole label (0 = road), or None when the source has no outlines
    xyz_mm     HxWx3 float32 point cloud in mm (NaN = no reading), or None
    flagged    True when the dataset's authors set the frame aside

Sources
-------
fan    Fan et al., "Pothole Detection Based on Disparity Transformation and Road Surface
       Modeling", IEEE T-IP 2019. 67 stereo frames in three sets, each with a disparity map,
       a 3D point cloud in mm and a pixel-level pothole label. MIT licence.
       https://github.com/ruirangerfan/stereo_pothole_datasets -> archive/fan_stereo/

Inspected and NOT added
-----------------------
PotholeDepth RGB-D (Zenodo 10.5281/zenodo.15773442; Butt et al., PLOS ONE 2026), in
archive/potholedepth/. The paper describes 25,000 photos with iPhone LiDAR depth. The files
are 25,013 frames of about 14 phone video clips of Lahore streets, each with a 224x320
8-BIT map stretched to the full 0-255 range in every frame checked (400 of 400). There is
no absolute scale in them, the sky carries values, and they look like the output of a
relative-depth network. Nothing in millimetres can be read from them, so they cannot train
or test a depth model here. The photos are used as unlabelled phone imagery in the style
bank. The shipped train/val/test lists split the clips frame by frame.
"""
import glob
import os

import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAN_DIR = os.path.join(PROJECT_DIR, "archive", "fan_stereo")
# README, section 5: "The following pothole detection results are not utilized for
# algorithm evaluation". Kept, flagged, and reported separately.
FAN_SET_ASIDE = {("dataset2", n) for n in (6, 7, 8, 13, 14, 15, 21, 22, 23, 37, 38, 39)}


def fan_frames():
    import scipy.io as sio
    for ds in ("dataset1", "dataset2", "dataset3"):
        for p in sorted(glob.glob(os.path.join(FAN_DIR, ds, "rgb", "*.png"))):
            n = os.path.splitext(os.path.basename(p))[0]
            xyz = sio.loadmat(os.path.join(FAN_DIR, ds, "ptcloud", n + ".mat"))["xyzPoints"].astype(np.float32)
            disp = sio.loadmat(os.path.join(FAN_DIR, ds, "disp", n + ".mat"))["disp"]
            ok = np.isfinite(xyz).all(-1) & (disp > 0) & (xyz[..., 2] > 0)
            yield {
                "key": f"fan_{ds[-1]}_{n}", "group": ds,
                "bgr": cv2.imread(p),
                "depth_mm": np.where(ok, xyz[..., 2], 0).astype(np.float32),
                "mask": (cv2.imread(os.path.join(FAN_DIR, ds, "label", n + ".png"), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8),
                "xyz_mm": np.where(ok[..., None], xyz, np.nan).astype(np.float32),
                "flagged": (ds, int(n)) in FAN_SET_ASIDE,
            }


RSRD_DIR = os.path.join(PROJECT_DIR, "RSRD-dense", "RSRD-dense")
RSRD_CALIB = os.path.join(PROJECT_DIR, "archive", "rsrd_devkit", "calibration_files")


def rsrd_drive_of():
    """
    Timestamp -> drive. The shipped test split is 300 frames drawn from the SAME drives as the
    training frames, 0.2 s apart, so it is not used as a test set here. Every frame is assigned
    to the drive it was recorded in (training folder names are drive start times; '-N-conti'
    clips belong to their parent drive), and splits are made by drive.
    """
    drives = {}
    for d in sorted(os.listdir(os.path.join(RSRD_DIR, "train"))):
        base = d.split("-conti")[0]
        base = base.rsplit("-", 1)[0] if base.count("-") == 6 else base      # drop the clip number
        drives[base] = drives.get(base, []) + [d]
    starts = sorted((b.replace("-", ""), b) for b in drives)                 # YYYYMMDDhhmmss
    return starts


def rsrd_frames(resolution="half"):
    """
    RSRD-dense (Tsinghua, arXiv 2310.02262, CC BY-NC): vehicle-mounted stereo, road ahead at 2-9 m,
    ground truth from several fused LiDAR scans projected into the left camera.

    Depth is computed from the DISPARITY map, not read from the depth map: both are stored as
    value x 256 in 16 bits, which quantises depth to 3.9 mm but disparity to 1/256 px, i.e.
    about 0.5 mm of depth at 4 m. depth_mm = B(mm) * fx / disparity, with the per-day calibration
    from the development kit (archive/rsrd_devkit). Only ~16% of pixels carry a reading.
    No pothole outlines: mask is None.
    """
    import pickle
    starts = rsrd_drive_of()
    suffix = "_half" if resolution == "half" else ""
    calib = {}
    for split in ("train", "test"):
        root = os.path.join(RSRD_DIR, split)
        folders = sorted(os.listdir(root)) if split == "train" else [""]
        for folder in folders:
            left = os.path.join(root, folder, "left" + suffix)
            for p in sorted(glob.glob(os.path.join(left, "*.jpg"))):
                stamp = os.path.splitext(os.path.basename(p))[0]
                t = stamp.split(".")[0]
                drive = [b for s, b in starts if s <= t][-1] if any(s <= t for s, _ in starts) else "unknown"
                day = t[:8]
                if day not in calib:
                    with open(os.path.join(RSRD_CALIB, f"calib_{day}{suffix}.pkl"), "rb") as f:
                        c = pickle.load(f)
                    calib[day] = (float(c["B"]) * float(c["K"][0, 0]), np.asarray(c["K"], dtype=np.float64))
                disp = cv2.imread(os.path.join(root, folder, "disparity" + suffix, stamp + ".png"), cv2.IMREAD_UNCHANGED)
                if disp is None:
                    continue
                d = disp.astype(np.float32) / 256.0
                depth = np.where(d > 0, calib[day][0] / np.maximum(d, 1e-6), 0).astype(np.float32)
                # K travels with the frame: this camera looks along the road, so relief is measured
                # perpendicular to it in 3D (build_relief_targets.relief_target_perp)
                yield {"key": f"rsrd_{stamp}", "group": drive, "bgr": cv2.imread(p), "depth_mm": depth,
                       "mask": None, "xyz_mm": None, "flagged": split == "test", "K": calib[day][1]}


SPLIT_SEED = 20261006


def rsrd_split(keys):
    """
    {key: "train" | "val" | "test"} by DRIVE. Whole drives are assigned, in a seeded random
    order, until test holds 20% of the frames and validation 10% (the PothRGBD rule).
    Frames of one drive are 0.2 s apart; a frame-level split would test on near-copies.
    """
    starts = rsrd_drive_of()

    def drive(k):
        t = k.split("_", 1)[1].split(".")[0]
        return [b for s, b in starts if s <= t][-1]
    groups = {k: drive(k) for k in keys}
    sizes = {}
    for g in groups.values():
        sizes[g] = sizes.get(g, 0) + 1
    order = np.random.default_rng(SPLIT_SEED).permutation(sorted(sizes))
    role, filled, total = {}, 0, len(keys)
    for g in order:
        role[g] = "test" if filled < 0.2 * total else "val" if filled < 0.3 * total else "train"
        filled += sizes[g]
    return {k: role[g] for k, g in groups.items()}


def native_bgr(key):
    """The photo at its own resolution, for a key written by one of the readers above."""
    source, rest = key.split("_", 1)
    if source == "fan":
        ds, n = rest.split("_")
        return cv2.imread(os.path.join(FAN_DIR, f"dataset{ds}", "rgb", n + ".png"))
    if source == "rsrd":
        hits = glob.glob(os.path.join(RSRD_DIR, "*", "*", "left", rest + ".jpg")) + \
            glob.glob(os.path.join(RSRD_DIR, "test", "left", rest + ".jpg"))
        return cv2.imread(hits[0]) if hits else None
    raise KeyError(key)


READERS = {"fan": fan_frames, "rsrd": rsrd_frames}
PRESENT = {"fan": os.path.isdir(FAN_DIR), "rsrd": os.path.isdir(RSRD_DIR) and os.path.isdir(RSRD_CALIB)}
