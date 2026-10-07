"""
Canonical registry of every image source in the project.

Why this exists
---------------
RoadLens now draws on six independent corpora collected on three continents with
four different annotation conventions and four incompatible licences. Without a
single place recording what came from where, three specific things go wrong, and
two of them already had:

1. **Split contamination.** `merged_dataset` was found to share 45 source photos
   between train and valid, 49 between train and test, and 11 between valid and
   test — 9-11% of each evaluation split had been seen in training. Every
   accuracy number computed on it is inflated by that, on top of the label
   circularity documented in Part 12 of MASTER_DOC.

2. **Ungrounded generalisation claims.** "Water recall 0.812" means something
   quite different if the evaluation set is entirely Korean urban roads — which
   it is. A number without its provenance is not a claim about the world.

3. **Licence leakage.** These range from CC BY 4.0 through academic-use-only to
   one dataset with no LICENCE file at all. Which corpus a figure came from
   determines what may be done with it.

Filename conventions, per source
--------------------------------
Provenance is partly recoverable from filenames, and each source does it
differently:

    merged_dataset     `rdd_*` / `kaggle_*` / `p600_*` / `gps_*` prefixes
    data1              Roboflow: `<stem>_jpg.rf.<32 hex>.jpg`
    puddle-seg         Roboflow, same pattern
    PothRGBD           `YYYYMMDD_HHMMSS_color_png.rf.<hex>.jpg`, depth keyed on
                       the timestamp prefix alone
    mendeley-water     `Pothole-NNN.jpg`, flat

`source_stem()` strips the Roboflow augmentation suffix so that augmented
variants of one photograph collapse to a single identity. That identity — not
the filename — is what any split must be grouped on.
"""
import os
import re
from typing import Dict, List, Optional

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Roboflow rewrites `foo.jpg` to `foo_jpg.rf.<md5>.jpg` and emits several
# augmented variants per source photo. data1 holds 780 files from just 300
# photographs; the water set holds 2419 from 1500.
_RF_SUFFIX = re.compile(r"_(jpe?g|png|bmp)\.rf\.[0-9a-f]+$", re.I)


class Source:
    def __init__(self, sid, name, root, region, annotation, licence,
                 url="", splits=None, notes="", key_rule="roboflow"):
        self.id = sid
        self.name = name
        self.root = root
        self.region = region
        self.annotation = annotation
        self.licence = licence
        self.url = url
        self.splits = splits or {}
        self.notes = notes
        self.key_rule = key_rule

    @property
    def path(self) -> str:
        return os.path.join(PROJECT_DIR, self.root)

    @property
    def present(self) -> bool:
        return os.path.isdir(self.path)

    def __repr__(self):
        return f"<Source {self.id} {'present' if self.present else 'ABSENT'}>"


REGISTRY: List[Source] = [
    Source(
        "kaggle-data1", "Kaggle pothole set (Roboflow export)",
        "data1", region="mixed / unspecified",
        annotation="YOLO bbox + seg polygons",
        licence="unspecified — inherited from the Roboflow export",
        splits={"train": "data1/train", "valid": "data1/valid"},
        notes="780 files from only 300 source photographs (2.6x augmentation). "
              "Splits verified clean: no source photo spans train and valid.",
    ),
    Source(
        "merged", "Merged corpus (Pothole-600 + RDD2022 + Kaggle)",
        "merged_dataset", region="Japan, India, Czechia, Norway, US + mixed",
        annotation="YOLO polygons, converted from Pascal VOC and masks",
        licence="mixed — inherits from each constituent source",
        splits={"train": "merged_dataset/train", "valid": "merged_dataset/valid",
                "test": "merged_dataset/test"},
        key_rule="prefix",
        notes="CONTAMINATED. 45 source photos shared between train and valid, "
              "49 between train and test, 11 between valid and test. Any metric "
              "computed on these splits is inflated. Re-split by source_stem "
              "before trusting a number from it.",
    ),
    Source(
        "pothrgbd", "PothRGBD — RGB + RealSense depth",
        os.path.join("archive", "PUBLIC POTHOLE DATASET"),
        region="Turkey (Intel RealSense D415, single operator)",
        annotation="YOLO-seg polygons + per-frame depth .npy in millimetres",
        licence="IEEE DataPort / Kaggle mirror — check before redistribution",
        url="https://www.kaggle.com/datasets/mahyeks/pothrgbd-rgb-and-depth-images-of-potholes",
        key_rule="timestamp",
        notes="The only source with MEASURED depth. Supplies the 1051 metric "
              "labels that break label circularity. One camera, one operator, "
              "close range — generalisation to phone imagery is unproven.",
    ),
    Source(
        "mendeley-water", "Annotated water-filled and dry potholes",
        "mendeley_water_filled_dataset",
        region="United Kingdom (mobile phone)",
        annotation="Pascal VOC XML + YOLO bbox — SINGLE class 'pothole'",
        licence="CC BY 4.0 (Mendeley Data)",
        url="https://data.mendeley.com/datasets/tp95cdvgm8/1",
        key_rule="flat",
        notes="Title promises water-filled AND dry, but the annotations do NOT "
              "distinguish them — all 1156 XML instances are class 'pothole'. "
              "Unusable as a water evaluation set without hand-labelling.",
    ),
    Source(
        "hanyang-water", "HanYang puddle segmentation",
        "puddle segmentation.v8i.yolov8",
        region="South Korea, urban roads",
        annotation="YOLO-seg polygons, single class 'puddle'",
        licence="CC BY 4.0",
        url="https://universe.roboflow.com/hanyang-university-bd2kb/puddle-segmentation",
        splits={"train": "puddle segmentation.v8i.yolov8/train",
                "valid": "puddle segmentation.v8i.yolov8/valid"},
        notes="2419 files from 1500 source photographs. Splits verified clean: "
              "0 source photos span train and valid. Supplies the Phase 4 "
              "calibration. All results from it describe KOREAN URBAN ROADS and "
              "should be reported that way.",
    ),
]

BY_ID: Dict[str, Source] = {s.id: s for s in REGISTRY}


def source_stem(filename: str) -> str:
    """
    Identity of the underlying PHOTOGRAPH, not the file.

    Strips the Roboflow augmentation suffix so `pic-1-_jpg.rf.49882c...jpg` and
    `pic-1-_jpg.rf.8d95dd...jpg` collapse to `pic-1-`. Any train/test split must
    group on this: two augmentations of one photo on opposite sides of a split
    is leakage, and it is invisible if you group on filename.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return _RF_SUFFIX.sub("", stem)


def pothrgbd_key(filename: str) -> Optional[str]:
    """PothRGBD pairs RGB, depth and label on a `YYYYMMDD_HHMMSS` prefix."""
    m = re.match(r"(\d{8}_\d{6})", os.path.basename(filename))
    return m.group(1) if m else None


def merged_origin(filename: str) -> str:
    """Which upstream corpus a merged_dataset file came from, by prefix."""
    b = os.path.basename(filename).lower()
    for pre, origin in (("rdd_", "RDD2022"), ("kaggle_", "Kaggle"),
                        ("p600_", "Pothole-600"), ("gps_", "GPS-tagged")):
        if b.startswith(pre):
            return origin
    return "unknown"


def identify(path: str) -> Optional[Source]:
    """Which registered source a path belongs to, longest root first."""
    ap = os.path.abspath(path)
    for s in sorted(REGISTRY, key=lambda x: -len(x.root)):
        if ap.startswith(os.path.abspath(s.path) + os.sep):
            return s
    return None


def grouped_split(paths, test_size=0.25, seed=20260909):
    """
    Split a list of image paths so no PHOTOGRAPH lands on both sides.

    Use this instead of `train_test_split` anywhere image paths are involved.
    Grouping on filename is not enough: `data1` holds 780 files from 300
    photographs and the water set 2419 from 1500, so a filename-level split puts
    augmented variants of the same photo in train and test simultaneously. The
    model then scores well on images it has effectively already seen.

    `merged_dataset` demonstrates the cost of not doing this — 45 photographs
    span its train/valid boundary and 49 span train/test, so 9-11% of each
    evaluation split was already in training.

    Returns (train_paths, test_paths).
    """
    import random

    groups = {}
    for p in paths:
        groups.setdefault(source_stem(p), []).append(p)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_test = max(1, int(round(len(keys) * test_size)))
    test_keys = set(keys[:n_test])

    train, test = [], []
    for k, ps in groups.items():
        (test if k in test_keys else train).extend(ps)
    return train, test


def audit_splits(split_paths: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Count photographs shared between named splits.

    `split_paths` maps a split name to its image paths. Returns a dict keyed
    "a|b" giving the number of shared photographs. An empty result means clean.

    Worth running before quoting any metric: a contaminated split inflates the
    number silently, and nothing in a training log reveals it.
    """
    stems = {name: {source_stem(p) for p in paths}
             for name, paths in split_paths.items()}
    out: Dict[str, int] = {}
    names = sorted(stems)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = stems[names[i]] & stems[names[j]]
            if ov:
                out[f"{names[i]}|{names[j]}"] = len(ov)
    return out


def describe() -> str:
    lines = ["RoadLens image sources", ""]
    for s in REGISTRY:
        mark = "present" if s.present else "ABSENT"
        lines.append(f"  [{mark:7s}] {s.id:16s} {s.name}")
        lines.append(f"{'':12s}region   {s.region}")
        lines.append(f"{'':12s}licence  {s.licence}")
        if s.notes:
            lines.append(f"{'':12s}note     {s.notes}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
