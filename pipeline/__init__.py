from .config import PipelineConfig
from .progress import ProgressReporter, STAGE_NAMES
from .model_resolution import (
    ModelResolution,
    resolve_player_model,
    resolve_ball_model,
    resolve_court_keypoint_model,
    describe_resolution,
)


def __getattr__(name: str):
    """Lazy-load the ML-heavy core symbols so that importing the pipeline
    package (e.g. to access PipelineConfig, ProgressReporter, or
    model_resolution) does NOT pull in ultralytics/torch at import time.
    run_analysis / resolve_device are only loaded when they're actually used."""
    if name in ("run_analysis", "resolve_device"):
        from .core import run_analysis, resolve_device  # noqa: PLC0415
        g = globals()
        g["run_analysis"] = run_analysis
        g["resolve_device"] = resolve_device
        return g[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
