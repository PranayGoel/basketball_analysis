"""
Shared dribble-event signal for the rule_violation_detector package.

Both DoubleDribbleDetector and TravelingDetector need to agree on what "a
dribble" is -- this module is the single source of truth for that so the two
heuristics can't silently drift apart on the definition. This is a best-effort
proxy for a real dribble bounce, not a validated measurement; see the package
docstring in rule_violation_detector/__init__.py for the broader limitations.
"""
import sys
sys.path.append('../')
from personal.basketball_analysis.utils.bbox_utils import measure_distance, get_center_of_bbox
from personal.basketball_analysis.rule_violation_detector.config import (
    DRIBBLE_MIN_AMPLITUDE_PX,
    DRIBBLE_MIN_CYCLE_FRAMES,
    DRIBBLE_MAX_CYCLE_FRAMES,
    WRIST_MIN_CONF,
)

LEFT_WRIST_INDEX = 9
RIGHT_WRIST_INDEX = 10


def _nearer_wrist_distance(ball_center, player_pose):
    """
    Return the distance from the ball center to whichever of the player's
    two wrist keypoints is nearer, or None if neither wrist keypoint meets
    WRIST_MIN_CONF this frame.

    Args:
        ball_center (tuple): (x, y) center of the ball this frame.
        player_pose (dict): This player's pose entry for this frame, with a
            "keypoints" list of [x, y, conf] in COCO order.

    Returns:
        float | None: The nearer-wrist distance, or None if untrustworthy.
    """
    keypoints = player_pose.get("keypoints", [])
    if len(keypoints) <= RIGHT_WRIST_INDEX:
        return None

    candidate_distances = []
    for wrist_index in (LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX):
        x, y, conf = keypoints[wrist_index]
        if conf >= WRIST_MIN_CONF:
            candidate_distances.append(measure_distance(ball_center, (x, y)))

    if not candidate_distances:
        return None

    return min(candidate_distances)


SEEKING_PEAK = "seeking_peak"      # tracking a rising run, hunting for its top
SEEKING_RETURN = "seeking_return"  # tracking a falling run from a confirmed peak


class _CycleTracker:
    """
    Tracks a single down-up (near -> far -> near) cycle for a continuous
    possession streak: distance grows by at least DRIBBLE_MIN_AMPLITUDE_PX
    from a local minimum (the ball leaving the hand) to a local maximum (a
    confirmed peak), then shrinks by at least DRIBBLE_MIN_AMPLITUDE_PX back
    down from that peak (the ball returning to the hand) -- that return point
    is the completed event, and the cycle length is measured from the local
    minimum that preceded the peak to that return frame. A fresh instance is
    created every time the possessor changes, so state never leaks across
    streaks.
    """

    def __init__(self):
        self.phase = SEEKING_PEAK
        # The most recent confirmed local minimum this rising/falling run is
        # measured from (starts as the first observed distance).
        self.valley_distance = None
        self.valley_frame = None
        # The highest (SEEKING_PEAK) or lowest (SEEKING_RETURN) distance seen
        # since entering the current phase -- the running candidate extremum.
        self.running_extremum_distance = None
        self.running_extremum_frame = None
        # Only meaningful during SEEKING_RETURN: the confirmed peak distance
        # the current falling run is measured against.
        self._confirmed_peak_distance = None

    def observe(self, frame_num, distance):
        """
        Feed one frame's ball-to-wrist distance into the cycle state.

        Returns:
            int | None: The frame_num a dribble event completed on (the
                "return to hand" frame), if this observation closed a valid
                down-up cycle; otherwise None.
        """
        if self.valley_distance is None:
            # First observation ever for this streak -- seed everything from here.
            self.valley_distance = distance
            self.valley_frame = frame_num
            self.running_extremum_distance = distance
            self.running_extremum_frame = frame_num
            return None

        if self.phase == SEEKING_PEAK:
            if distance >= self.running_extremum_distance:
                self.running_extremum_distance = distance
                self.running_extremum_frame = frame_num
                return None

            if self.running_extremum_distance - distance >= DRIBBLE_MIN_AMPLITUDE_PX:
                # Confirmed a peak (the running max) -- the ball has moved far
                # enough from the valley. Now watch for the shrink back down.
                peak_distance = self.running_extremum_distance
                self.phase = SEEKING_RETURN
                self.running_extremum_distance = distance
                self.running_extremum_frame = frame_num
                self._confirmed_peak_distance = peak_distance
            return None

        # phase == SEEKING_RETURN
        if distance <= self.running_extremum_distance:
            self.running_extremum_distance = distance
            self.running_extremum_frame = frame_num

            if self._confirmed_peak_distance - distance >= DRIBBLE_MIN_AMPLITUDE_PX:
                # Shrunk back far enough from the peak -- cycle complete.
                cycle_length = frame_num - self.valley_frame
                return_frame = frame_num

                # This return point becomes the new valley for the next cycle.
                self.valley_distance = distance
                self.valley_frame = frame_num
                self.phase = SEEKING_PEAK
                self.running_extremum_distance = distance
                self.running_extremum_frame = frame_num

                if DRIBBLE_MIN_CYCLE_FRAMES <= cycle_length <= DRIBBLE_MAX_CYCLE_FRAMES:
                    return return_frame
            return None

        # Distance rose again before confirming the return -- the previous
        # running minimum since the peak becomes the new valley reference,
        # and we resume hunting for an even higher peak from here.
        self.valley_distance = self.running_extremum_distance
        self.valley_frame = self.running_extremum_frame
        self.phase = SEEKING_PEAK
        self.running_extremum_distance = distance
        self.running_extremum_frame = frame_num
        return None


def detect_dribble_events(ball_tracks, possessor_ids, pose_tracks):
    """
    A 'dribble event' = while a single player_id continuously possesses the
    ball, the ball's distance to that player's NEARER wrist keypoint completes
    a down-up cycle: distance grows past DRIBBLE_MIN_AMPLITUDE_PX from a local
    minimum, then shrinks back below it, within a frame-count window bounded
    by [DRIBBLE_MIN_CYCLE_FRAMES, DRIBBLE_MAX_CYCLE_FRAMES]. Wrist keypoints
    below WRIST_MIN_CONF are treated as missing (not trusted).

    This is a heuristic proxy for a real dribble bounce, built from noisy 2D
    keypoints and ball-detection positions -- it will miss dribbles and may
    occasionally invent one from a coincidental hand-wave. It is not a
    validated measurement of ball-handling.

    Args:
        ball_tracks: BallTracker.get_object_tracks_with_kalman() output.
        possessor_ids: BallAquisitionDetector.detect_ball_possession() output (list[int], -1 = none).
        pose_tracks: PoseEstimator.get_object_poses() output.

    Returns:
        list[dict]: [{"player_id": int, "frame": int}, ...] one entry per
            detected dribble event, "frame" = the cycle's "return to hand" frame.
    """
    num_frames = len(possessor_ids)
    events = []

    current_possessor = None
    tracker = None

    for frame_num in range(num_frames):
        possessor_id = possessor_ids[frame_num]

        if possessor_id != current_possessor:
            current_possessor = possessor_id
            tracker = _CycleTracker() if possessor_id != -1 else None

        if possessor_id == -1 or tracker is None:
            continue

        ball_info = ball_tracks[frame_num].get(1, {}) if frame_num < len(ball_tracks) else {}
        ball_bbox = ball_info.get("bbox")
        if not ball_bbox:
            continue

        frame_poses = pose_tracks[frame_num] if frame_num < len(pose_tracks) else {}
        player_pose = frame_poses.get(possessor_id)
        if player_pose is None:
            continue

        ball_center = get_center_of_bbox(ball_bbox)
        distance = _nearer_wrist_distance(ball_center, player_pose)
        if distance is None:
            continue

        completed_frame = tracker.observe(frame_num, distance)
        if completed_frame is not None:
            events.append({"player_id": possessor_id, "frame": completed_frame})

    return events
