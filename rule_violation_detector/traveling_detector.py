"""
Heuristic traveling detector -- see rule_violation_detector/__init__.py for
the package-level limitations this proxy is subject to.
"""
import sys
sys.path.append('../')
from rule_violation_detector.config import (
    ANKLE_MIN_CONF,
    STEP_MIN_VERTICAL_VELOCITY_DELTA,
    LEGAL_STEP_COUNT_THRESHOLD,
    MIN_VIOLATION_FRAME_SPAN,
)
from rule_violation_detector.dribble_event_detector import detect_dribble_events

LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16


class _FootPlantCounter:
    """
    Counts foot-plant events (local maxima in ankle-y, i.e. the foot hitting
    the ground) for one player across consecutive frames, using either ankle.
    Low-confidence ankle keypoints are treated as missing and don't count.
    """

    def __init__(self):
        self._previous_y = {LEFT_ANKLE_INDEX: None, RIGHT_ANKLE_INDEX: None}
        self._previous_velocity = {LEFT_ANKLE_INDEX: None, RIGHT_ANKLE_INDEX: None}

    def observe(self, player_pose):
        """
        Feed one frame's pose into the per-ankle velocity tracking.

        Returns:
            int: The number of foot-plant events detected on this frame
                (0, 1, or 2 -- both ankles could plant on the same frame,
                though that's rare for a walking gait).
        """
        plants_this_frame = 0
        keypoints = player_pose.get("keypoints", []) if player_pose else []

        for ankle_index in (LEFT_ANKLE_INDEX, RIGHT_ANKLE_INDEX):
            if len(keypoints) <= ankle_index:
                self._previous_y[ankle_index] = None
                self._previous_velocity[ankle_index] = None
                continue

            x, y, conf = keypoints[ankle_index]
            if conf < ANKLE_MIN_CONF:
                self._previous_y[ankle_index] = None
                self._previous_velocity[ankle_index] = None
                continue

            previous_y = self._previous_y[ankle_index]
            if previous_y is None:
                self._previous_y[ankle_index] = y
                self._previous_velocity[ankle_index] = None
                continue

            velocity = y - previous_y
            previous_velocity = self._previous_velocity[ankle_index]

            if previous_velocity is not None:
                # A foot-plant = the ankle was moving down (positive velocity,
                # since image y grows downward) and now has swung to moving up
                # (or stopped hard), with enough magnitude to not be jitter.
                if (
                    previous_velocity > 0
                    and velocity <= 0
                    and (previous_velocity - velocity) >= STEP_MIN_VERTICAL_VELOCITY_DELTA
                ):
                    plants_this_frame += 1

            self._previous_y[ankle_index] = y
            self._previous_velocity[ankle_index] = velocity

        return plants_this_frame


class _WindowState:
    """Per-possession-streak bookkeeping for the step-count window."""

    __slots__ = ("start_frame", "step_count", "exceeded", "foot_plants")

    def __init__(self, start_frame):
        self.start_frame = start_frame
        self.step_count = 0
        self.exceeded = False
        self.foot_plants = _FootPlantCounter()


class TravelingDetector:
    """
    HEURISTIC, best-effort flag -- see package docstring on limitations.
    Counts foot-plant events (ankle vertical-velocity direction changes,
    i.e. local maxima in ankle-y = the foot hitting the ground) during
    continuous ball possession WITHOUT an intervening dribble event.
    Exceeding LEGAL_STEP_COUNT_THRESHOLD flags the window.

    KNOWN LIMITATION: this cannot determine pivot foot or the exact "gather"
    moment real officiating depends on. It answers a cruder question --
    "did this player take an unusually large number of steps while holding
    the ball without dribbling?" -- which correlates with traveling but is
    not equivalent to it.

    NOTE ON SIGNATURE: detecting "an intervening dribble event" requires the
    same ball-position signal DoubleDribbleDetector uses (via the shared
    detect_dribble_events()), so detect() takes ball_tracks in addition to
    possessor_ids and pose_tracks -- without it this detector could not see
    dribbles at all and its core reset behavior would be unimplementable.
    """

    def detect(self, possessor_ids, pose_tracks, ball_tracks):
        """
        Args:
            possessor_ids: BallAquisitionDetector.detect_ball_possession() output.
            pose_tracks: PoseEstimator.get_object_poses() output.
            ball_tracks: BallTracker.get_object_tracks_with_kalman() output --
                needed to compute the shared dribble-event signal that resets
                the step-count window (see class docstring).

        Returns:
            list[dict]: [{"violation_type": "traveling", "player_id": int,
                "start_frame": int, "end_frame": int, "confidence": "heuristic"}, ...]
        """
        dribble_events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)
        dribble_frames_by_player = {}
        for event in dribble_events:
            dribble_frames_by_player.setdefault(event["player_id"], set()).add(event["frame"])

        num_frames = len(possessor_ids)
        violations = []

        current_possessor = None
        window = None

        def close_window(end_frame):
            if window is None:
                return
            span = end_frame - window.start_frame + 1
            if window.exceeded and span >= MIN_VIOLATION_FRAME_SPAN:
                violations.append({
                    "violation_type": "traveling",
                    "player_id": current_possessor,
                    "start_frame": window.start_frame,
                    "end_frame": end_frame,
                    "confidence": "heuristic",
                })

        for frame_num in range(num_frames):
            possessor_id = possessor_ids[frame_num]

            if possessor_id != current_possessor:
                # Streak boundary -- flush any pending violation for the
                # player who just lost possession before switching state.
                close_window(frame_num - 1)
                current_possessor = possessor_id
                window = _WindowState(start_frame=frame_num) if possessor_id != -1 else None

            if possessor_id == -1 or window is None:
                continue

            frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}
            player_pose = frame_poses.get(possessor_id)

            player_dribbled_this_frame = frame_num in dribble_frames_by_player.get(possessor_id, ())
            if player_dribbled_this_frame:
                # Dribbling legally resets the step allowance -- walking while
                # dribbling is normal. Re-scope the window so a later flagged
                # violation covers only the actual no-dribble walking span.
                window = _WindowState(start_frame=frame_num + 1)
                continue

            plants = window.foot_plants.observe(player_pose)
            if plants:
                window.step_count += plants
                if window.step_count > LEGAL_STEP_COUNT_THRESHOLD:
                    window.exceeded = True

        # Flush whatever streak was still open when the video ended.
        close_window(num_frames - 1)

        return violations
