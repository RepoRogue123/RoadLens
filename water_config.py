"""
Single source of truth for the Phase 4 water-detection decision parameters.

Before this module every weight, threshold and correction constant was a literal
inside water_detection.detect_water, hand-tuned against four images. The
docstring there still records the provenance honestly:

    "pic-52 was falsely classified DRY (16.9%) but has subtle moisture"
    "pic-65 was falsely classified WATER (47.1%) but is dry gravel in sunlight"
    "This alone fixes pic-65 false positive"

Constants fitted to two named images are not a model, they are a memory of those
two images. Extracting them here does not make them right — it makes them
visible, overridable, and fittable, which is what scripts/eval_water_puddle1000.py
needs in order to replace them with values derived from a labelled set.

Once that script has run against Puddle-1000 it writes
ml_results/phase4_calibration/water_params.json, and that file takes precedence,
so the numbers actually in force are always traceable to the run that produced
them.
"""
import json
import os
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_JSON = os.path.join(
    _HERE, "ml_results", "phase4_calibration", "water_params.json"
)

# ── Uncalibrated defaults ────────────────────────────────────────────────
# Reasoned from optics rather than fitted. Each maps to a physical property of
# water, which is why a wrong answer can be traced to a responsible cue — but
# the relative magnitudes below are judgement, not measurement.
DEFAULT_CUE_WEIGHTS: Dict[str, float] = {
    # ── always available ──
    "edge_density": 0.25,   # water is near-featureless (~3%), gravel is not (~35%)
    "gradient": 0.20,       # Sobel magnitude ratio, interior vs road ring
    "specular": 0.15,       # spatial clustering of glints, not mere brightness
    "color": 0.15,          # blue-channel ratio; the reflected sky is Rayleigh-blue
    "saturation": 0.10,     # low, uniform saturation, gated on darkness
    # ── requires depth_map + curvature_features ──
    "inconsistency": 0.15,  # boundary says crater, interior depth says flat
    # ── requires a CLIPSeg prior map (semantic_water.py) ──
    "semantic": 0.15,       # open-vocabulary "puddle" response, interior vs ring
    # ── requires an Intrinsic residual map (intrinsic_cues.py) ──
    # Held low deliberately. Measured on 13 regions, the positive-residual
    # interior/ring ratio has median 0.712 with only 23% above 1.0, and the
    # cue's direction is supported by just two labelled images. At weight 0.20
    # it measurably HARMED the ensemble (pic-52, the one image labelled wet,
    # fell from 0.234 to 0.210 when it was added). It stays in at low weight so
    # the Puddle-1000 fit can raise it on evidence or drop it on evidence.
    "residual": 0.10,       # non-diffuse energy — Fresnel reflection made measurable
}

# ── Cue 8 response curve ─────────────────────────────────────────────────
# Interior/ring ratio of positive residual, mapped to [0, 1]. The original
# 1.0 -> 2.5 range was invented and saturated 77% of regions at zero, which is
# how a cue ends up contributing nothing but dilution. These values come from
# the observed distribution (min 0.193, median 0.712, max 1.181) and are still
# provisional — they are a scale taken from data, not a fitted decision boundary.
DEFAULT_RESIDUAL_RATIO_LOW = 0.85    # at or below -> 0.0
DEFAULT_RESIDUAL_RATIO_HIGH = 1.25   # at or above -> 1.0

# Cues present on every call. The rest are added only when their input is
# supplied, and the ensemble renormalises over whatever is available.
BASE_CUES = ("edge_density", "gradient", "specular", "color", "saturation")

# Decision threshold on the combined probability.
DEFAULT_DECISION_THRESHOLD = 0.35

# Confidence banding, applied above the decision threshold.
DEFAULT_CONFIDENCE_BANDS = {"high": 0.65, "medium": 0.45, "low": 0.35}

# ── Non-linear corrections ───────────────────────────────────────────────
# These are the most overfitted part of the module. Each was introduced to fix
# one named image. They are retained so behaviour is unchanged until a fit
# exists, and they are the first thing the Puddle-1000 evaluation should test
# for removal.
DEFAULT_CORRECTIONS = {
    # Both strongest cues agree -> compress toward the high end.
    "agreement_boost_threshold": 0.6,
    "agreement_boost_scale": 0.65,
    "agreement_boost_offset": 0.35,
    # Rough surface -> hard-cap the probability. Introduced for pic-65.
    "rough_cap_hard_edge": 0.15,
    "rough_cap_hard_value": 0.25,
    "rough_cap_soft_edge": 0.30,
    "rough_cap_soft_value": 0.40,
    # High interior gradient -> scale down. Introduced for pic-65.
    "gradient_penalty_edge": 0.15,
    "gradient_penalty_scale": 0.6,
}

# Whether the non-linear corrections are applied at all. The evaluation harness
# flips this to measure how much of the module's behaviour they account for.
DEFAULT_APPLY_CORRECTIONS = True

# ── Cue 7 / cue 8 rollout gates ──────────────────────────────────────────
# Cue 7 (CLIPSeg) is ON. The HanYang evaluation earned it: fitted on the
# 2,119-image train split and tested on the disjoint 300, the calibrated
# ensemble cut missed water from 244 to 83 of 442, and `semantic` carries the
# largest coefficient. The shipped water_params.json is the 6-cue fit that
# REQUIRES this cue — with it off, detect_water must refuse the calibration and
# fall back to the hand-set configuration, so the measured numbers would
# describe a model the API never runs.
#
# Cue 8 (Intrinsic) stays OFF. Dropping it cost 7 extra misses out of 442 for a
# 16x speed-up, and its licence is academic-only. The 7-cue calibration is kept
# at water_params_7cue_backup.json for offline batch analysis.
#
# scripts/eval_water_puddle1000.py passes the maps in explicitly regardless of
# these flags, so evaluation is unaffected by them. Where CLIPSeg cannot be
# imported (e.g. a container without `transformers`), the cue abstains and
# detect_water reports combination="legacy_fallback_missing_cues".
#
# Cost when enabled: one forward pass per IMAGE (not per pothole) — roughly
# 0.2 s for CLIPSeg and 2-4 s for Intrinsic on the 4070.
ENABLE_SEMANTIC_CUE = True
ENABLE_RESIDUAL_CUE = False

# ── Learned water segmenter (DECIDES since 2026-10-02, on human labels) ──────
# A YOLOv8n-seg trained on HanYang puddle outlines (scripts/train_water_yolo.py).
# On the held-out HanYang valid split — the same 707 regions the logistic model
# was scored on — it reaches precision 0.998 / recall 0.925 (33 missed water
# regions, 1 false alarm) against 0.924 / 0.796 (90 missed, 29 false) for the
# served 6-cue logistic. It also needs no `transformers`, so unlike CLIPSeg it
# runs in the deployed container.
#
# On PothRGBD potholes, which are mostly dry, both detectors flag the same share
# (17 of 173) and by eye most flags are dry, brown, dirt-filled potholes: HanYang
# puddles are often muddy brown. There are no pothole-water labels, so that rate
# is not measured, only inspected. Water never changes severity.
#
# HanYang regions ARE water outlines, but in the pipeline the region is the
# POTHOLE outline and water often fills only its centre, so the "> 50%" rule that
# worked on HanYang misses partly filled potholes. The rule for potholes was
# settled on 142 hand-labelled potholes (53 water, 89 dry; three labellers, two
# per pothole, kappa 0.70, disputes resolved by review; packs/water_v1,
# scripts/eval_pothole_water.py), under a switching rule written before labelling:
#
#                                   missed water   false alarms
#     cue ensemble (6-cue logistic)       22             11
#     segmenter, cross-fitted             13             15      <- the honest estimate
#     segmenter, coverage > 0.1           11             15      <- served; in-sample
#
# Known cost: on dry, dirt-filled potholes (PothRGBD: 4 water of 45) it raises 9
# false alarms against the ensemble's 5 — it reads brown soil as muddy water.
# Water never changes severity; it only raises the hazard flag.
# The ensemble still runs and is reported as `ensemble_probability`.
ENABLE_WATER_SEGMENTER = True           # run it and report coverage
WATER_SEGMENTER_DECIDES = True          # False restores the cue ensemble as the decision
WATER_SEGMENTER_WEIGHTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "yolo-segmentation", "model", "water_best.pt")
WATER_SEGMENTER_CONF = 0.25
WATER_COVERAGE_THRESHOLD = 0.1          # share of the pothole outline marked as water
# How MUCH of the pothole is covered, not how sure the call is: precision was flat
# (0.73-0.75) across thresholds from 0.1 to 0.5 on the labelled potholes.
WATER_COVERAGE_BANDS = {"high": 0.5, "medium": 0.3, "low": 0.1}


def _load() -> Dict[str, Any]:
    if os.path.isfile(CALIBRATION_JSON):
        try:
            with open(CALIBRATION_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_CAL = _load()

CUE_WEIGHTS: Dict[str, float] = {
    **DEFAULT_CUE_WEIGHTS,
    **(_CAL.get("cue_weights") or {}),
}
DECISION_THRESHOLD: float = float(
    _CAL.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
)
CONFIDENCE_BANDS: Dict[str, float] = {
    **DEFAULT_CONFIDENCE_BANDS,
    **(_CAL.get("confidence_bands") or {}),
}
CORRECTIONS: Dict[str, float] = {
    **DEFAULT_CORRECTIONS,
    **(_CAL.get("corrections") or {}),
}
RESIDUAL_RATIO_LOW: float = float(
    _CAL.get("residual_ratio_low", DEFAULT_RESIDUAL_RATIO_LOW)
)
RESIDUAL_RATIO_HIGH: float = float(
    _CAL.get("residual_ratio_high", DEFAULT_RESIDUAL_RATIO_HIGH)
)
APPLY_CORRECTIONS: bool = bool(
    _CAL.get("apply_corrections", DEFAULT_APPLY_CORRECTIONS)
)

IS_CALIBRATED: bool = bool(_CAL)

# ── Signed (logistic) combination ────────────────────────────────────────
# The legacy ensemble computes sum(w*s) / sum(w) with non-negative weights, so
# it can only express "this cue argues FOR water, more or less strongly". It
# cannot express a cue that argues AGAINST water.
#
# That is not a hypothetical limitation. Fitted against real labelled water
# (HanYang, 707 regions), `saturation` scores a coefficient of -2.427 — arguing
# against water almost as strongly as `edge_density` argues for it — and
# `gradient` contributes +0.333 against a hand-set weight of 0.20. Under the
# non-negative sum both cues actively fight the ensemble, and recall suffers:
# 0.448 as shipped.
#
# The physical reason is worth keeping: both cues assume water is smooth and
# washed-out. True for a puddle reflecting a uniform overcast sky; false for one
# reflecting a structured urban scene, because the reflection carries the
# structure of the reflected world. Road photography is mostly the second case.
#
# When a calibration file supplies `logistic`, detect_water switches to
#     z = intercept + sum(coef_i * score_i)      p = 1 / (1 + exp(-z))
# which admits negative coefficients and is exactly the form the fitter
# produces. Without it, behaviour is unchanged.
_LOGIT = _CAL.get("logistic") or {}
LOGISTIC_COEF: Dict[str, float] = {
    k: float(v) for k, v in (_LOGIT.get("coef") or {}).items()
}
LOGISTIC_INTERCEPT: float = float(_LOGIT.get("intercept", 0.0))
USE_LOGISTIC: bool = bool(LOGISTIC_COEF)


def describe() -> str:
    """One-line provenance string, safe to log or surface in the API."""
    if IS_CALIBRATED:
        return (
            f"calibrated (n={_CAL.get('n_labelled', '?')}, "
            f"fitted {_CAL.get('fitted_at', 'unknown date')}, "
            f"set={_CAL.get('dataset', 'unknown')}): "
            f"threshold={DECISION_THRESHOLD:.3f} "
            f"corrections={'on' if APPLY_CORRECTIONS else 'off'}"
        )
    return (
        "UNCALIBRATED hand-set defaults, tuned against 4 images "
        f"(threshold={DECISION_THRESHOLD}, corrections="
        f"{'on' if APPLY_CORRECTIONS else 'off'}). "
        "Run scripts/eval_water_puddle1000.py --fit to replace."
    )


if __name__ == "__main__":
    print(describe())
    print()
    for name, w in CUE_WEIGHTS.items():
        tag = "base" if name in BASE_CUES else "optional"
        print(f"  {name:14s} {w:.3f}  ({tag})")
