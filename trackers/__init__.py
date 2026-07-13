"""Lazy re-exports so that importing the trackers package (e.g. to access
BallTracker or kalman_ball_tracker submodules) does NOT eagerly load
player_tracker.py (which imports ultralytics/torch at module level).
Code that does `from personal.basketball_analysis.trackers import PlayerTracker`
still works -- the import is just deferred until first access."""


def __getattr__(name: str):
    if name == "PlayerTracker":
        from .player_tracker import PlayerTracker  # noqa: PLC0415
        globals()["PlayerTracker"] = PlayerTracker
        return PlayerTracker
    if name == "BallTracker":
        from .ball_tracker import BallTracker  # noqa: PLC0415
        globals()["BallTracker"] = BallTracker
        return BallTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
