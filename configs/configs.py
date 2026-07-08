import os

# Model weight paths are anchored to the repo root (where this configs/ package
# lives), NOT the calling process's cwd. This matters in practice: main.py has
# always been run from the repo root by convention, but webapp/backend's arq
# worker runs from webapp/backend/ -- a bare relative 'models/player_detector.pt'
# would silently resolve to webapp/backend/models/player_detector.pt there
# instead of the real repo-root models/ directory. Caught via a live end-to-end
# test of the web platform, not a unit test (there was nothing to unit-test --
# the bug only exists as a function of *where a process happens to be launched
# from*, which no in-process test exercises).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _repo_path(*parts):
    return os.path.join(_REPO_ROOT, *parts)


STUBS_DEFAULT_PATH = 'stubs'
PLAYER_DETECTOR_PATH = _repo_path('models', 'player_detector.pt')
BALL_DETECTOR_PATH = _repo_path('models', 'ball_detector_model.pt')
COURT_KEYPOINT_DETECTOR_PATH = _repo_path('models', 'court_keypoint_detector.pt')
OUTPUT_VIDEO_PATH = 'output_videos/output_video.mp4'

# Inference defaults (overridable via CLI flags in main.py)
DEVICE = 'auto'          # 'auto' resolves to cuda -> mps -> cpu; or force 'cpu'/'cuda'/'mps'
BATCH_SIZE = 20          # frames per YOLO prediction batch

# Per-model confidence thresholds. These used to be one shared CONF_THRESHOLD,
# but the three models have different accuracy profiles: the ball is small,
# fast-moving, and often motion-blurred (needs a lower threshold to even
# surface real-but-uncertain detections -- safe now that the Kalman gate in
# BallTracker.get_object_tracks_with_kalman rejects motion-implausible false
# positives instead of relying on a high confidence floor to do that alone).
PLAYER_CONF_THRESHOLD = 0.5
BALL_CONF_THRESHOLD = 0.3
COURT_KEYPOINT_CONF_THRESHOLD = 0.4

# Backward-compat alias: some callers may still import CONF_THRESHOLD as a
# single global default. Kept equal to the player threshold (the previous
# shared default) so nothing importing it breaks; main.py's --conf flag, when
# explicitly passed, still overrides all three models at once.
CONF_THRESHOLD = PLAYER_CONF_THRESHOLD

# Test-time augmentation (TTA) is enabled only for the ball model: it's the
# smallest/hardest-to-detect object and benefits most from multi-scale
# inference; the player and court-keypoint models are already reliable enough
# that TTA's added inference cost isn't worth paying on every frame.
PLAYER_AUGMENT = False
BALL_AUGMENT = True
COURT_KEYPOINT_AUGMENT = False

# Half precision (FP16) is opt-in and off by default -- MPS fp16 maturity
# varies by torch version, so don't assume it's free without benchmarking on
# the actual target machine first (see scripts/benchmark_detection.py).
USE_HALF_PRECISION = False

# Kalman ball-tracker tuning (see trackers/kalman_ball_tracker.py).
BALL_KF_PROCESS_NOISE = 5.0
BALL_KF_MEASUREMENT_NOISE = 10.0
BALL_KF_MAX_REASONABLE_DISTANCE_PER_FRAME = 60.0

# ByteTrack occlusion-handling tuning for PlayerTracker (see
# trackers/player_tracker.py). Defaults per the sv.ByteTrack() constructor are
# track_activation_threshold=0.25, lost_track_buffer=30,
# minimum_matching_threshold=0.8, minimum_consecutive_frames=1 -- tuned here to
# extend occlusion tolerance and suppress spurious short-lived tracks.
PLAYER_TRACK_ACTIVATION_THRESHOLD = 0.4
PLAYER_LOST_TRACK_BUFFER = 60
PLAYER_MINIMUM_MATCHING_THRESHOLD = 0.75
PLAYER_MINIMUM_CONSECUTIVE_FRAMES = 2