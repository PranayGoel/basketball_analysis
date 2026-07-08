"""
Model resolution: decides which weight file (and expected class name) each of
the 3 detection models actually uses, in priority order:

  1. An explicit env var override (PLAYER_DETECTOR_PATH_OVERRIDE, etc.) --
     lets a fine-tuning adoption step (scripts/finetune_model.py), or a manual
     choice, point the live pipeline at any weights file without touching
     configs.py or copying anything into the canonical models/ path.
  2. The real custom-trained weights at the canonical models/ path, if present.
  3. (player/ball only) A generic COCO-pretrained YOLO fallback that
     self-downloads from Ultralytics' own GitHub release assets -- no
     corporate-network friction, unlike the Google-Drive-hosted custom
     weights -- with the matching COCO class name substituted in.
  4. (court_keypoint only) No fallback exists: this is a bespoke task with no
     COCO equivalent. Returns None, signaling degraded-mode operation --
     pipeline/core.py skips court-keypoint-dependent stages (tactical view,
     real-world-units speed/distance) entirely rather than pretending.

Every resolution is meant to be printed/logged by the caller (see
`describe_resolution`), never silently substituted -- this project's own
culture (the LLM section's "honest gap, not a claim," the pose section's
"worth a human reviewing this clip") extends here: if you're not looking at
the real trained models, you should know it from the first line of output.
"""

import os
from dataclasses import dataclass
from typing import Optional

from personal.basketball_analysis.configs import PLAYER_DETECTOR_PATH, BALL_DETECTOR_PATH, COURT_KEYPOINT_DETECTOR_PATH

COCO_FALLBACK_MODEL = 'yolo11n.pt'
COCO_PERSON_CLASS = 'person'
COCO_SPORTS_BALL_CLASS = 'sports ball'

PLAYER_OVERRIDE_ENV_VAR = 'PLAYER_DETECTOR_PATH_OVERRIDE'
BALL_OVERRIDE_ENV_VAR = 'BALL_DETECTOR_PATH_OVERRIDE'
COURT_KEYPOINT_OVERRIDE_ENV_VAR = 'COURT_KEYPOINT_DETECTOR_PATH_OVERRIDE'

FALLBACK_ACCURACY_NOTE = (
    "accuracy will be materially lower than the real trained model -- "
    "generic COCO classes don't distinguish players from refs/spectators, "
    "or a basketball from any other round ball"
)


@dataclass
class ModelResolution:
    """
    weights_path: what to pass to YOLO(...).
    class_name: the exact string the tracker matches detections against
        (e.g. 'Player' for the custom model, 'person' for the COCO fallback).
        Unused/empty for the court-keypoint model, which has no class-name
        concept (CourtKeypointDetector reads .keypoints directly, no filter).
    source: "override" | "custom" | "fallback" -- for the startup log line.
    """
    weights_path: str
    class_name: str
    source: str


def _resolve_detector(override_env_var, custom_path, custom_class_name,
                       fallback_weights, fallback_class_name):
    override = os.environ.get(override_env_var)
    if override:
        return ModelResolution(weights_path=override, class_name=custom_class_name, source="override")
    if os.path.exists(custom_path):
        return ModelResolution(weights_path=custom_path, class_name=custom_class_name, source="custom")
    return ModelResolution(weights_path=fallback_weights, class_name=fallback_class_name, source="fallback")


def resolve_player_model() -> ModelResolution:
    """Priority: PLAYER_DETECTOR_PATH_OVERRIDE env var -> real weights at
    PLAYER_DETECTOR_PATH -> generic COCO yolo11n.pt ('person' class)."""
    return _resolve_detector(
        PLAYER_OVERRIDE_ENV_VAR, PLAYER_DETECTOR_PATH, 'Player',
        COCO_FALLBACK_MODEL, COCO_PERSON_CLASS,
    )


def resolve_ball_model() -> ModelResolution:
    """Priority: BALL_DETECTOR_PATH_OVERRIDE env var -> real weights at
    BALL_DETECTOR_PATH -> generic COCO yolo11n.pt ('sports ball' class)."""
    return _resolve_detector(
        BALL_OVERRIDE_ENV_VAR, BALL_DETECTOR_PATH, 'Ball',
        COCO_FALLBACK_MODEL, COCO_SPORTS_BALL_CLASS,
    )


def resolve_court_keypoint_model() -> Optional[ModelResolution]:
    """
    Priority: COURT_KEYPOINT_DETECTOR_PATH_OVERRIDE env var -> real weights at
    COURT_KEYPOINT_DETECTOR_PATH -> None (degraded mode -- no COCO-equivalent
    fallback exists for this bespoke task; see pipeline/core.py for exactly
    what running in degraded mode skips).

    Returns:
        ModelResolution, or None if no weights are available at all.
    """
    override = os.environ.get(COURT_KEYPOINT_OVERRIDE_ENV_VAR)
    if override:
        return ModelResolution(weights_path=override, class_name='', source="override")
    if os.path.exists(COURT_KEYPOINT_DETECTOR_PATH):
        return ModelResolution(weights_path=COURT_KEYPOINT_DETECTOR_PATH, class_name='', source="custom")
    return None


def describe_resolution(model_label: str, resolution: Optional[ModelResolution]) -> str:
    """
    Formats one loud, honest startup log line for a resolved model. Callers
    (pipeline/core.py) should always print this -- never resolve silently.
    """
    if resolution is None:
        return f"[model] {model_label}: DEGRADED -- no weights available, related features disabled"

    line = f"[model] {model_label}: source={resolution.source} weights={resolution.weights_path}"
    if resolution.class_name:
        line += f" class='{resolution.class_name}'"
    if resolution.source == "fallback":
        line += f" -- {FALLBACK_ACCURACY_NOTE}"
    return line
