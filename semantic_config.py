"""
Single source of truth for the Phase 3 semantic-override decision parameters.

Before this module the variance-ratio threshold 0.9 was duplicated in four
places (api.py, scripts/visualize_semantic.py, scripts/compare_failure_modes.py,
scripts/phase3_diagnostics.py), so changing it in one place silently left the
others disagreeing.

The values below are defaults. Once scripts/calibrate_semantic.py has been run
against a labelled set it writes ml_results/phase3_calibration/threshold.json,
and that file takes precedence — so the number actually in force is always
traceable to the run that produced it.
"""
import json
import os
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_JSON = os.path.join(
    _HERE, "ml_results", "phase3_calibration", "threshold.json"
)

# ── Uncalibrated defaults ────────────────────────────────────────────────
# 0.9 is the original hand-set value, retained so behaviour is unchanged until
# a calibration file exists. Diagnostics show it is NOT trustworthy: see
# ml_results/phase3_diagnostics/ — one pothole under three augmentations
# straddles it (0.84 / 0.92 / 0.95).
DEFAULT_DOWNGRADE_RATIO = 0.9

# Two-way rule: a ratio well ABOVE 1 means the interior is markedly more
# heterogeneous than the road, i.e. positive evidence of a real crater. Used to
# raise an under-reported verdict. Deliberately conservative until calibrated.
DEFAULT_UPGRADE_RATIO = 1.35

# Abstention: DINOv2 gives a 14x14 patch grid, so a small pothole may occupy
# only a handful of patches. A variance over fewer than this many samples is
# too thin to vote on — the witness should decline rather than guess.
DEFAULT_MIN_PATCHES = 8


def _load() -> Dict[str, Any]:
    if os.path.isfile(CALIBRATION_JSON):
        try:
            with open(CALIBRATION_JSON, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_CAL = _load()

DOWNGRADE_RATIO: float = float(_CAL.get("downgrade_ratio", DEFAULT_DOWNGRADE_RATIO))
UPGRADE_RATIO: float = float(_CAL.get("upgrade_ratio", DEFAULT_UPGRADE_RATIO))
MIN_PATCHES: int = int(_CAL.get("min_patches", DEFAULT_MIN_PATCHES))

#  Which variance ratio the rule consumes. "corrected" is the ring-local,
#  size-matched statistic; "raw" is the original whole-scene comparison kept
#  for backwards comparability.
RATIO_KEY: str = _CAL.get("ratio_key", "dinov2_ratio_corrected")

IS_CALIBRATED: bool = bool(_CAL)


def describe() -> str:
    """One-line provenance string, safe to log or surface in the API."""
    if IS_CALIBRATED:
        return (
            f"calibrated (n={_CAL.get('n_labelled', '?')}, "
            f"fitted {_CAL.get('fitted_at', 'unknown date')}): "
            f"downgrade<{DOWNGRADE_RATIO:.3f} upgrade>{UPGRADE_RATIO:.3f} "
            f"min_patches={MIN_PATCHES} key={RATIO_KEY}"
        )
    return (
        f"UNCALIBRATED hand-set defaults: downgrade<{DOWNGRADE_RATIO} "
        f"upgrade>{UPGRADE_RATIO} min_patches={MIN_PATCHES} key={RATIO_KEY}"
    )


if __name__ == "__main__":
    print(describe())
