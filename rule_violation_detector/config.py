"""
Named threshold constants for the rule_violation_detector package.

These are starting-point defaults chosen to be plausible for ~30fps broadcast
footage, NOT validated parameters. They have not been tuned against a labeled
dataset of real double-dribble/traveling clips. Expect to adjust them after
watching how the detectors perform on actual sample videos -- treat every
value here as a first guess, not a calibrated constant.
"""

# --- Dribble-cycle detection (rule_violation_detector.dribble_event_detector) ---
DRIBBLE_MIN_AMPLITUDE_PX = 40        # min ball-to-wrist distance swing to count as a bounce
DRIBBLE_MIN_CYCLE_FRAMES = 4         # fastest plausible dribble cycle (reject noise)
DRIBBLE_MAX_CYCLE_FRAMES = 45        # slowest plausible dribble cycle (~1.5s @ 30fps)
WRIST_MIN_CONF = 0.3                 # ignore wrist keypoints below this confidence

# --- Double dribble (rule_violation_detector.double_dribble_detector) ---
PICKUP_PROXIMITY_PX = 35             # ball must be within this of BOTH wrists to count as "held"
PICKUP_MIN_CONSECUTIVE_FRAMES = 5    # both-wrist proximity must hold this many frames to count as a deliberate pickup

# --- Traveling (rule_violation_detector.traveling_detector) ---
ANKLE_MIN_CONF = 0.3
STEP_MIN_VERTICAL_VELOCITY_DELTA = 3.0   # px/frame velocity-sign-change threshold for a foot-plant event
LEGAL_STEP_COUNT_THRESHOLD = 3            # 0..THIS many steps = legal; more = flagged (real rule is ~2 after "gather", but the gather moment isn't reliably detectable from 2D keypoints -- see module docstrings)
MIN_VIOLATION_FRAME_SPAN = 3               # suppress violations spanning fewer frames than this (keypoint-glitch noise floor)
