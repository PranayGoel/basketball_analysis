"""
PipelineConfig: the single, explicit set of knobs run_analysis() accepts.

Carries every option introduced across the detection-accuracy and pose-model
work (per-model confidence, TTA/half precision, Kalman tuning stays internal
to BallTracker's own defaults, --pose) so both the CLI (main.py) and the web
backend build this from their own argument sources (argparse vs. an API
request body) and then call the identical run_analysis() function -- one
source of pipeline truth, not two.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    video_path: str
    output_video_path: str = "output_videos/output_video.mp4"
    report_path: Optional[str] = None
    stub_dir: str = "stubs"
    device: str = "auto"
    use_stubs: bool = True

    # None means "use that model's own tuned default from configs.py" -- see
    # main.py's _resolve_conf() for the exact precedence when a CLI caller
    # also has a global override; the web backend can just pass explicit
    # per-model values directly since it has no equivalent "global override"
    # concept in its request shape.
    player_conf: Optional[float] = None
    ball_conf: Optional[float] = None
    court_conf: Optional[float] = None

    half: bool = False
    batch_size: int = 20

    # Pose estimation + heuristic rule-violation detection (double dribble,
    # traveling) -- off by default, real added compute cost. See
    # pose_estimator/ and rule_violation_detector/.
    pose: bool = False
    pose_model: str = "yolo11n-pose.pt"
