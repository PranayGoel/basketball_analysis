"""
rule_violation_detector -- best-effort HEURISTIC flags for double dribble and
traveling, built from ball position, ball possession, and pose keypoints
already computed elsewhere in this pipeline.

IMPORTANT LIMITATIONS -- read before trusting any output from this package:

These detectors are best-effort proxies, NOT validated officiating calls.
Real basketball rules for both violations hinge on details this pipeline
cannot reliably determine from noisy 2D keypoints on broadcast footage:

- Double dribble legally depends on the exact moment a player re-establishes
  control after a dribble ends (a clean "possession start"), which isn't
  cleanly separable from ordinary ball-handling using bbox/keypoint proximity
  alone.
- Traveling legally depends on identifying the player's PIVOT FOOT and the
  precise "gather" moment (when the dribble officially ends and the step
  count starts). Neither is reliably derivable from 2D pose keypoints; this
  package instead counts foot-plant events during possession without an
  intervening dribble, which correlates with traveling but is not the same
  question a referee is answering.

Every output from this package should be read as "this clip is worth a human
reviewing," never as a ruling. That's consistent with the honesty this
codebase already practices elsewhere -- an untested code path being called
"an honest gap, not a claim" is the same spirit these detectors aim for.
"""
from .double_dribble_detector import DoubleDribbleDetector
from .traveling_detector import TravelingDetector
from .dribble_event_detector import detect_dribble_events
