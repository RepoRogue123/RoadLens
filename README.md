<div align="center">
  <img src="docs/assets/hero.svg" alt="RoadLens" width="800"/>

  <h1>RoadLens: Pothole Detection and Measured Severity</h1>

  <p><em>Finds potholes in a road photo, estimates how deep each one is in millimetres, and turns that into a safety-weighted severity.</em></p>

  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white&style=for-the-badge)](https://fastapi.tiangolo.com/)
  [![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black&style=for-the-badge)](https://react.dev/)
  [![YOLOv8](https://img.shields.io/badge/YOLOv8-Segmentation-111111?style=for-the-badge)](https://docs.ultralytics.com/)
</div>

---

## What it does

For each pothole in a photo, RoadLens:

1. **Outlines it** with a YOLOv8 segmenter trained on true pothole outlines plus RDD2022 and Mendeley boxes converted to outlines with SAM 2.
2. **Estimates its depth below the road**, in millimetres, with a Depth-Anything-V2 network fine-tuned on measured depth (RealSense and LiDAR). Small potholes are read from a close-up crop.
3. **Assigns a severity** (Shallow / Moderate / Deep) with cut-offs chosen so that under-reporting a hazard costs three times as much as over-reporting it, and gives a likely range.
4. **Checks for water**, which hides a pothole's floor, with a learned water segmenter.
5. **Warns when the photo is unlike the measured training photos**, using MoGe-3 geometry, because the depth is then an extrapolation.

## Results on held-out data

All numbers are on data the models never trained on. Splits are made by capture session or drive, never by frame.

| | Result |
|---|---|
| Depth error, 208 measured potholes in unseen PothRGBD sessions | 8.0 mm average (hand-drawn outlines), 8.3 mm (segmenter outlines); correlation 0.76; 80% within 13.3 mm |
| Depth below the road, 705 frames from unseen RSRD drives | 6.6 mm per pixel (median over frames) |
| Segmenter, potholes found (overlap ≥ 0.5) | PothRGBD 91.3%, Pothole-600 83.6%, Kaggle 71.1% |
| Segmenter on an unseen country (RDD2022 Japan) | 22.2% of pothole boxes found; 7.3% of pothole-free frames get a false outline |
| Water, 142 potholes labelled wet or dry by two people each | 11 of 53 wet potholes missed, 15 false alarms of 89 dry |

**Published approaches, rebuilt and run on the same 207 held-out potholes:**

| Approach | Depth error |
|---|---|
| Segment, then zero-shot metric depth (Depth-Anything-V2, outdoor model) | 123.5 mm |
| Segment, then zero-shot MoGe-3 | 39.9 mm |
| Same network fine-tuned on plain depth from the camera | 17.3 mm |
| Guess the median depth for every pothole | 13.6 mm |
| **RoadLens (network fine-tuned on depth below the road)** | **8.0 mm** |

---

## Repository layout

```text
Vision-Based-Pothole-Detection/
├── api.py                    # FastAPI service: /healthz, /analyze, /insights/*
├── segmentation.py           # pothole segmenter (YOLOv8-seg), picks the newest weights found
├── relief_depth.py           # depth network: mm below the road, per pixel; close-up reading
├── metric_severity.py        # depth -> severity, range, out-of-range check
├── metric_features.py        # road-plane fitting and geometry features
├── moge_backend.py           # MoGe-3 wrapper (optional)
├── water_detection.py        # water decision (segmenter + cue ensemble)
├── water_segmenter.py        # learned water segmenter
├── water_config.py           # water settings
├── semantic_water.py         # CLIPSeg water cue
├── foundation_features.py    # DINOv2 crater-or-stain check (advisory only)
├── semantic_config.py        # its settings
├── mask_refine.py            # SAM 2 outline refinement and box-to-outline
├── dataset_registry.py       # every dataset: source, licence, split rules
├── features.py, classifier.py, ml_classifier.py, inference.py, main.py
│                             # legacy feature pipeline and classifier vote (shown, not used for severity)
├── shape_from_shading.py, adverse_conditions.py, intrinsic_cues.py, temporal_analysis.py
├── road_segment_analysis.py  # batch report for a folder of photos
├── scripts/                  # data preparation, training and evaluation (see below)
├── tests/                    # pipeline and water-detection tests
├── web-ui/                   # React + Vite dashboard
├── yolo-segmentation/model/  # segmenter and water-segmenter weights (included)
├── ml_models/                # legacy classifier models; ml_models/metric/ = depth models
├── docs/assets/              # README images
├── requirements.txt, Dockerfile, .env.example
└── (local only, not in git) datasets, Depth-Anything-V2/, archive/, ml_results/
```

**`scripts/` by purpose**

| Purpose | Scripts |
|---|---|
| Measured labels | `pothrgbd_metric_labels.py`, `build_metric_trainset.py`, `build_relief_targets.py`, `depth_sources.py` |
| Segmenter data and training | `build_yolo_seg_v2.py`, `build_corpus_manifest.py`, `boxes_to_polygons.py`, `build_yolo_seg_v3.py`, `train_yolo_seg_v2.py`, `convert_rdd2022.py`, `convert_pothole600.py` |
| Depth training | `train_relief_depth.py`, `train_metric_regressor.py`, `build_style_bank.py`, `build_unlabelled_crops.py` |
| Water | `train_water_yolo.py`, `train_water_yolo_v2.py`, `eval_pothole_water.py`, `eval_water_puddle1000.py` |
| Evaluation | `eval_relief_depth.py` (also installs depth models), `eval_relief_sources.py`, `eval_relief_pothole600.py`, `eval_zoom_inference.py`, `eval_yolo_seg.py`, `eval_yolo_rdd.py`, `eval_yolo_downstream.py`, `compare_served_severity.py` |
| Comparison with published approaches | `eval_zero_shot_metric.py`, `train_severity_classifier.py`, `make_counterexamples.py` |
| Team labelling | `build_annotation_pack.py`, `annotate_pack.py`, `merge_annotations.py`, `label_pothole_water.py` |
| Dataset audit | `build_provenance_index.py`, `deduplicate.py`, `verify_dataset.py`, `dataset_summary.py` |

---

## Data layout

Datasets are not in the repository. Download each one you need and place it exactly here (paths are relative to the repository root):

| Dataset | Place at | Used for | Source |
|---|---|---|---|
| PothRGBD (RGB + RealSense depth) | `archive/PUBLIC POTHOLE DATASET/{images,depths,labels}/` | depth labels, depth training and testing, segmenter | [Kaggle](https://www.kaggle.com/datasets/mahyeks/pothrgbd-rgb-and-depth-images-of-potholes) |
| Kaggle pothole segmentation | `data1/{train,valid}/{images,labels}/` | segmenter | Roboflow export |
| Pothole-600 | `pothole600/{training,validation,testing}/{rgb,label,tdisp}/` | segmenter, second-camera shape test | [Pothole-600](https://sites.google.com/view/pothole-600) |
| RDD2022 | `rdd_temp/<Country>/<Country>/{train,test}/...` (unzipped per country) | segmenter (boxes outlined by SAM 2), negatives | [RDD2022](https://github.com/sekilab/RoadDamageDetector) |
| Mendeley water-filled and dry potholes | `mendeley_water_filled_dataset/An Annotated Water-Filled, and Dry Potholes Dataset for Deep Learning Applications/{IMG,XML}/` | segmenter | [Mendeley Data](https://data.mendeley.com/datasets/tp95cdvgm8/1) |
| HanYang puddle segmentation | `puddle segmentation.v8i.yolov8/{train,valid}/` | water segmenter | [Roboflow](https://universe.roboflow.com/hanyang-university-bd2kb/puddle-segmentation) |
| RSRD-dense | `RSRD-dense/RSRD-dense/{train,test}/` | depth training (vehicle camera), testing | [RSRD](https://thu-rsxd.com/rsrd/) (download in a browser) |
| RSRD calibration | `archive/rsrd_devkit/` | RSRD depth from disparity | `git clone https://github.com/ztsrxh/RSRD_dev_toolkit archive/rsrd_devkit` |
| Fan stereo pothole sets | `archive/fan_stereo/dataset{1,2,3}/` | second-camera depth test | `git clone https://github.com/ruirangerfan/stereo_pothole_datasets archive/fan_stereo` |

Everything the scripts generate (training sets, caches, runs, results) is written under `archive/` and `ml_results/`, both ignored by git.

---

## Model weights

| Weights | Location | In the repository |
|---|---|---|
| Pothole segmenter v3 (and the older `best.pt`) | `yolo-segmentation/model/best_v3.pt` | yes |
| Water segmenter | `yolo-segmentation/model/water_best.pt` | yes |
| Depth regressor (fallback, needs MoGe-3) | `ml_models/metric/depth_regressor.joblib` + `_meta.json` | yes |
| Depth networks (two, averaged) | `ml_models/metric/relief_vits_1.pth`, `relief_vits_2.pth` + `relief_meta.json` | **no** (about 100 MB each): train them, below |
| Depth-Anything-V2 code and base checkpoint | `Depth-Anything-V2/` and `Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth` | no: `git clone https://github.com/DepthAnything/Depth-Anything-V2` and download the ViT-S checkpoint |
| MoGe-3 | downloaded automatically on first use | no |

Without the depth networks the app falls back to the MoGe regressor (if MoGe is installed), and otherwise to the legacy classifier vote, and says which one it used.

---

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
copy .env.example .env

cd web-ui
copy .env.example .env      # VITE_API_BASE_URL=http://localhost:8000
npm install
```

## Running

```powershell
python api.py               # backend on http://localhost:8000
cd web-ui; npm run dev      # dashboard
```

Settings (environment variables):

| Variable | Default | Effect |
|---|---|---|
| `ROADLENS_SEVERITY_MODE` | `metric` | `legacy` restores the old classifier vote as the headline severity |
| `ROADLENS_DEPTH_SOURCE` | `relief` | `regressor` uses the MoGe regressor even when the depth networks exist |
| `ROADLENS_ZOOM_SMALL` | `1` | `0` turns off the close-up reading of small potholes |
| `ROADLENS_SEG_WEIGHTS` | newest found | path to other segmenter weights |

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/healthz` | Service and model status |
| `POST` | `/analyze` | Multipart image upload. Returns each pothole's outline, depth in mm with a range, severity, water check, module status and overlay images |
| `GET` | `/insights/summary` | Latest evaluation metrics for the dashboard |
| `GET` | `/insights/files/{file}` | Evaluation figures for the dashboard |

---

## Reproducing the models

Run from the repository root, in this order. Each script's docstring explains its inputs, outputs and decision rules.

```powershell
# 1. Measured depth labels and the fallback regressor (needs MoGe-3)
python scripts/pothrgbd_metric_labels.py
python scripts/build_metric_trainset.py
python scripts/train_metric_regressor.py

# 2. Pothole segmenter v3
python scripts/build_yolo_seg_v2.py
python scripts/build_corpus_manifest.py
python scripts/boxes_to_polygons.py --gate --convert
python scripts/build_yolo_seg_v3.py
python scripts/train_yolo_seg_v2.py --data archive/yolo_seg_v3/data_full.yaml --name seg_v3_full --out yolo-segmentation/model/best_v3.pt

# 3. Depth networks (RSRD targets first), then install the averaged pair
python scripts/build_relief_targets.py --source rsrd
python scripts/train_relief_depth.py --epochs 28 --name vits_v3 --plane-invariant
python scripts/train_relief_depth.py --epochs 40 --name vits_v8 --exact-plane --rsrd 1.0 --batch 6 --rsrd-batch 2
python scripts/eval_relief_depth.py --weights archive/relief_runs/vits_v8/best.pth,archive/relief_runs/vits_v3/best.pth --install

# 4. Water segmenter
python scripts/train_water_yolo.py
```

Evaluation and the comparison with published approaches:

```powershell
python scripts/eval_yolo_seg.py yolo-segmentation/model/best_v3.pt@0.35
python scripts/eval_yolo_rdd.py yolo-segmentation/model/best_v3.pt@0.35
python scripts/build_relief_targets.py --source fan
python scripts/eval_relief_sources.py --source fan
python scripts/eval_relief_sources.py --source rsrd --split test
python scripts/make_counterexamples.py
python scripts/eval_zero_shot_metric.py
python scripts/train_severity_classifier.py
```

## Tests

```powershell
python tests/test_pipeline.py
python tests/test_water_detection.py
```

---

## Licences of the data

PothRGBD (IEEE DataPort / Kaggle), RSRD (CC BY-NC: non-commercial), HanYang puddles and Mendeley (CC BY 4.0), Fan stereo sets (MIT), RDD2022 (see its repository). Check each licence before redistributing data or models trained on it.
