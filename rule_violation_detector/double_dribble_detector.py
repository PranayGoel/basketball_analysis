"""
Heuristic double-dribble detector -- see rule_violation_detector/__init__.py
for the package-level limitations this proxy is subject to.
"""
from personal.basketball_analysis.utils.bbox_utils import measure_distance, get_center_of_bbox
from personal.basketball_analysis.rule_violation_detector.config import (
    PICKUP_PROXIMITY_PX,
    PICKUP_MIN_CONSECUTIVE_FRAMES,
    WRIST_MIN_CONF,
)
from personal.basketball_analysis.rule_violation_detector.dribble_event_detector import (
    detect_dribble_events,
    LEFT_WRIST_INDEX,
    RIGHT_WRIST_INDEX,
)

NOT_POSSESSING = "NOT_POSSESSING"
DRIBBLING = "DRIBBLING"
HELD = "HELD"


def _both_wrists_holding_ball(ball_center, player_pose):
    """
    True if the ball center is within PICKUP_PROXIMITY_PX of BOTH of the
    player's wrist keypoints this frame (both wrists must also meet
    WRIST_MIN_CONF -- a low-confidence wrist can't confirm a two-hand hold).

    Args:
        ball_center (tuple): (x, y) center of the ball this frame.
        player_pose (dict): This player's pose entry for this frame.

    Returns:
        bool
    """
    keypoints = player_pose.get("keypoints", [])
    if len(keypoints) <= RIGHT_WRIST_INDEX:
        return False

    for wrist_index in (LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX):
        x, y, conf = keypoints[wrist_index]
        if conf < WRIST_MIN_CONF:
            return False
        if measure_distance(ball_center, (x, y)) > PICKUP_PROXIMITY_PX:
            return False

    return True


class DoubleDribbleDetector:
    """
    HEURISTIC, best-effort flag -- see package docstring on limitations.
    Detects: dribble -> two-hand pickup -> dribble again, within one
    continuous (uninterrupted) ball-possession streak.
    """

    def detect(self, ball_tracks, possessor_ids, pose_tracks):
        """
        Args:
            ball_tracks: BallTracker.get_object_tracks_with_kalman() output.
            possessor_ids: BallAquisitionDetector.detect_ball_possession() output.
            pose_tracks: PoseEstimator.get_object_poses() output.

        Returns:
            list[dict]: [{"violation_type": "double_dribble", "player_id": int,
                "start_frame": int, "end_frame": int, "confidence": "heuristic"}, ...]
        """
        dribble_events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)
        dribble_frames_by_player = {}
        for event in dribble_events:
            dribble_frames_by_player.setdefault(event["player_id"], set()).add(event["frame"])

        num_frames = len(possessor_ids)
        violations = []

        current_possessor = None
        state = NOT_POSSESSING
        held_streak_start_frame = None
        held_since_frame = None
        consecutive_hold_frames = 0

        def reset_streak_state():
            nonlocal state, held_streak_start_frame, held_since_frame, consecutive_hold_frames
            state = NOT_POSSESSING
            held_streak_start_frame = None
            held_since_frame = None
            consecutive_hold_frames = 0

        for frame_num in range(num_frames):
            possessor_id = possessor_ids[frame_num]

            if possessor_id != current_possessor:
                current_possessor = possessor_id
                reset_streak_state()

            if possessor_id == -1:
                continue

            player_dribbled_this_frame = frame_num in dribble_frames_by_player.get(possessor_id, ())

            if state == NOT_POSSESSING:
                if player_dribbled_this_frame:
                    state = DRIBBLING
                continue

            if state == DRIBBLING:
                ball_info = ball_tracks[frame_num].get(1, {}) if frame_num < len(ball_tracks) else {}
                ball_bbox = ball_info.get("bbox")
                frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}
                player_pose = frame_poses.get(possessor_id)

                is_held_this_frame = bool(
                    ball_bbox and player_pose is not None
                    and _both_wrists_holding_ball(get_center_of_bbox(ball_bbox), player_pose)
                )

                if is_held_this_frame:
                    if consecutive_hold_frames == 0:
                        held_streak_start_frame = frame_num
                    consecutive_hold_frames += 1
                    if consecutive_hold_frames >= PICKUP_MIN_CONSECUTIVE_FRAMES:
                        state = HELD
                        held_since_frame = held_streak_start_frame
                else:
                    consecutive_hold_frames = 0
                    held_streak_start_frame = None
                continue

            # state == HELD
            if player_dribbled_this_frame:
                violations.append({
                    "violation_type": "double_dribble",
                    "player_id": possessor_id,
                    "start_frame": held_since_frame,
                    "end_frame": frame_num,
                    "confidence": "heuristic",
                })
                # Reset so a later, independent dribble->pickup->dribble sequence
                # for the same player can still be flagged separately.
                reset_streak_state()

        return violations
