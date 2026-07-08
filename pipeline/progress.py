"""
ProgressReporter: a thin, shared stage-boundary instrumentation layer.

The CLI's --profile flag and the web backend's job-progress SSE stream both
need to know "which pipeline stage is running now" -- rather than each
inventing its own progress-tracking logic (which would drift out of sync with
run_analysis()'s actual steps over time), both wrap the same optional
callback contract: callback(stage_name: str, stage_index: int, total_stages:
int, status: "running" | "done").
"""

from contextlib import contextmanager

# Order matches run_analysis()'s actual execution order exactly -- keep these
# in sync if a stage is added/removed/reordered in pipeline/core.py.
STAGE_NAMES = [
    "read_video",
    "player_tracking",
    "ball_tracking",
    "court_keypoints",
    "team_assignment",
    "ball_acquisition",
    "pose_violations",
    "pass_interception",
    "tactical_view",
    "speed_distance",
    "drawing",
    "save_video",
    "report",
]


class ProgressReporter:
    """
    Wraps an optional callback and tracks which of STAGE_NAMES is active.

    Intentionally reports (stage_index, total_stages) rather than a fabricated
    weighted percentage -- different stages take wildly different wall-clock
    time depending on video length/hardware, and a made-up per-stage weight
    would be a guess dressed up as precision. "Stage 4 of 13: team_assignment"
    is honest; a bogus "38% complete" is not.
    """

    def __init__(self, callback=None, stage_names=STAGE_NAMES):
        self._callback = callback
        self._stage_names = list(stage_names)
        self._total = len(self._stage_names)

    @contextmanager
    def stage(self, name):
        """Usage: `with reporter.stage("read_video"): ...`."""
        index = self._stage_names.index(name) if name in self._stage_names else -1
        self._emit(name, index, "running")
        try:
            yield
        finally:
            self._emit(name, index, "done")

    def _emit(self, name, index, status):
        if self._callback is not None:
            self._callback(name, index, self._total, status)
