import unittest
from personal.basketball_analysis.rule_violation_detector.traveling_detector import TravelingDetector


ANKLE_X = 0.0


def _pose_with_ankle_y(y, conf=1.0, wrist_conf=0.0):
    """A pose with the left ankle at the given y (right ankle low-confidence,
    ignored) and wrists low-confidence by default so dribble detection never
    accidentally fires from these fixtures unless a test explicitly wants it to."""
    keypoints = [[0.0, 0.0, 1.0]] * 17
    keypoints[9] = [0.0, 0.0, wrist_conf]    # left wrist
    keypoints[10] = [0.0, 0.0, wrist_conf]   # right wrist
    keypoints[15] = [ANKLE_X, y, conf]        # left ankle
    keypoints[16] = [ANKLE_X, 0.0, 0.0]       # right ankle, unused/low-confidence
    return {"keypoints": keypoints, "bbox": [0, 0, 50, 100]}


def _footstep_ankle_ys(base=0.0):
    """
    Ankle-y values producing exactly ONE foot-plant event, verified by hand:
    y:        [base, base, base+10, base+2]
    velocity:      [0,      +10,       -8]
    The transition from velocity +10 (frame 2) to -8 (frame 3) is a plant:
    prev_velocity=10>0, velocity=-8<=0, delta=10-(-8)=18 >= STEP_MIN_VERTICAL_VELOCITY_DELTA=3.
    Returns a list of 4 y-values (frames 0-3 relative to `base`'s starting frame).
    """
    return [base, base, base + 10, base + 2]


def _empty_ball_tracks(count):
    return [{} for _ in range(count)]


class TestTravelingDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TravelingDetector()

    def _run(self, possessor_ids, pose_tracks, ball_tracks=None):
        if ball_tracks is None:
            ball_tracks = _empty_ball_tracks(len(possessor_ids))
        return self.detector.detect(possessor_ids, pose_tracks, ball_tracks)

    def test_two_steps_no_dribble_no_violation(self):
        # Two footsteps (2 <= LEGAL_STEP_COUNT_THRESHOLD=3) while possessing,
        # no dribble at all -- legal, no violation.
        step_1 = _footstep_ankle_ys(base=0)     # frames 0-3, 1 plant at frame 3
        step_2 = _footstep_ankle_ys(base=2)     # frames 4-7, 1 plant at frame 7
        ankle_ys = step_1 + step_2
        possessor_ids = [1] * len(ankle_ys)
        pose_tracks = [{1: _pose_with_ankle_y(y)} for y in ankle_ys]

        violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(violations, [])

    def test_four_steps_no_dribble_flags_traveling(self):
        # Four footsteps (> LEGAL_STEP_COUNT_THRESHOLD=3) with no intervening
        # dribble -- flags traveling.
        ankle_ys = (
            _footstep_ankle_ys(base=0)
            + _footstep_ankle_ys(base=2)
            + _footstep_ankle_ys(base=4)
            + _footstep_ankle_ys(base=6)
        )
        possessor_ids = [1] * len(ankle_ys)
        pose_tracks = [{1: _pose_with_ankle_y(y)} for y in ankle_ys]

        violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual(violation["violation_type"], "traveling")
        self.assertEqual(violation["player_id"], 1)
        self.assertEqual(violation["confidence"], "heuristic")
        self.assertEqual(violation["start_frame"], 0)
        self.assertEqual(violation["end_frame"], len(ankle_ys) - 1)

    def test_dribble_resets_step_count_no_violation_if_each_leg_legal(self):
        # Two steps, a dribble (legally resets the step allowance), then two
        # more steps -- each leg stays <= LEGAL_STEP_COUNT_THRESHOLD=3, so no
        # violation should fire for either leg.
        from personal.basketball_analysis.tests.test_dribble_event_detector import _ball_track, _pose_with_wrist

        leg_1_ankle_ys = _footstep_ankle_ys(base=0) + _footstep_ankle_ys(base=2)  # 8 frames, 2 plants
        dribble_ball_positions = [(d, 0) for d in (5, 20, 60, 90, 60, 20, 5, 5, 5, 5)]  # 10 frames
        leg_2_ankle_ys = _footstep_ankle_ys(base=4) + _footstep_ankle_ys(base=6)  # 8 frames, 2 plants

        num_leg1 = len(leg_1_ankle_ys)
        num_dribble = len(dribble_ball_positions)
        num_leg2 = len(leg_2_ankle_ys)
        total_frames = num_leg1 + num_dribble + num_leg2

        possessor_ids = [1] * total_frames

        pose_tracks = []
        for y in leg_1_ankle_ys:
            pose_tracks.append({1: _pose_with_ankle_y(y)})
        for _ in range(num_dribble):
            # During the dribble segment, wrist keypoints must be trustworthy
            # (high confidence) so detect_dribble_events can see the bounce;
            # ankle stays flat (no stepping) so no foot-plants are miscounted.
            pose = _pose_with_ankle_y(leg_1_ankle_ys[-1], wrist_conf=1.0)
            pose["keypoints"][9] = [0.0, 0.0, 1.0]
            pose["keypoints"][10] = [0.0, 0.0, 1.0]
            pose_tracks.append({1: pose})
        for y in leg_2_ankle_ys:
            pose_tracks.append({1: _pose_with_ankle_y(y)})

        ball_tracks = (
            _empty_ball_tracks(num_leg1)
            + [_ball_track(cx, cy) for cx, cy in dribble_ball_positions]
            + _empty_ball_tracks(num_leg2)
        )

        violations = self.detector.detect(possessor_ids, pose_tracks, ball_tracks)

        self.assertEqual(violations, [])

    def test_dribble_resets_but_second_leg_still_travels(self):
        # Two legal steps, a dribble (resets), then FOUR steps -- the second
        # leg alone exceeds LEGAL_STEP_COUNT_THRESHOLD=3, so exactly one
        # violation fires, and its start_frame is re-scoped to just after
        # the dribble (not the original stream start), covering only the
        # second leg's walking span.
        leg_1_ankle_ys = _footstep_ankle_ys(base=0) + _footstep_ankle_ys(base=2)  # 8 frames, legal
        dribble_ball_positions = [(d, 0) for d in (5, 20, 60, 90, 60, 20, 5, 5, 5, 5)]  # 10 frames
        leg_2_ankle_ys = (
            _footstep_ankle_ys(base=4)
            + _footstep_ankle_ys(base=6)
            + _footstep_ankle_ys(base=8)
            + _footstep_ankle_ys(base=10)
        )  # 16 frames, 4 plants -- travels

        num_leg1 = len(leg_1_ankle_ys)
        num_dribble = len(dribble_ball_positions)
        num_leg2 = len(leg_2_ankle_ys)
        total_frames = num_leg1 + num_dribble + num_leg2

        possessor_ids = [1] * total_frames

        pose_tracks = []
        for y in leg_1_ankle_ys:
            pose_tracks.append({1: _pose_with_ankle_y(y)})
        for _ in range(num_dribble):
            pose = _pose_with_ankle_y(leg_1_ankle_ys[-1])
            pose["keypoints"][9] = [0.0, 0.0, 1.0]
            pose["keypoints"][10] = [0.0, 0.0, 1.0]
            pose_tracks.append({1: pose})
        for y in leg_2_ankle_ys:
            pose_tracks.append({1: _pose_with_ankle_y(y)})

        ball_tracks = (
            _empty_ball_tracks(num_leg1)
            + [{1: {"bbox": [cx - 5, cy - 5, cx + 5, cy + 5], "source": "detection"}} for cx, cy in dribble_ball_positions]
            + _empty_ball_tracks(num_leg2)
        )

        violations = self.detector.detect(possessor_ids, pose_tracks, ball_tracks)

        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual(violation["player_id"], 1)
        # The dribble event fires at some frame within [num_leg1, num_leg1+num_dribble),
        # and the re-scoped window starts the frame right after it -- well
        # past leg_1's frames, and strictly within/after the dribble segment.
        self.assertGreaterEqual(violation["start_frame"], num_leg1)
        self.assertEqual(violation["end_frame"], total_frames - 1)

    def test_possessor_change_flushes_pending_violation(self):
        # Four steps (travels) for player 1, then possession switches to
        # player 2 -- the violation must be emitted at the streak boundary,
        # not only if the video happened to end there.
        ankle_ys = (
            _footstep_ankle_ys(base=0)
            + _footstep_ankle_ys(base=2)
            + _footstep_ankle_ys(base=4)
            + _footstep_ankle_ys(base=6)
        )
        trailing_frames = [0.0, 0.0, 0.0]  # a few more frames after the switch, for player 2

        possessor_ids = [1] * len(ankle_ys) + [2] * len(trailing_frames)
        pose_tracks = [{1: _pose_with_ankle_y(y)} for y in ankle_ys]
        pose_tracks += [{2: _pose_with_ankle_y(y)} for y in trailing_frames]

        violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["player_id"], 1)
        self.assertEqual(violations[0]["end_frame"], len(ankle_ys) - 1)

    def test_low_confidence_ankle_not_counted_as_footplants(self):
        # The exact same y-pattern as a violation-triggering sequence, but
        # every ankle keypoint is below ANKLE_MIN_CONF=0.3 -- no foot-plants
        # should register at all, so no violation.
        ankle_ys = (
            _footstep_ankle_ys(base=0)
            + _footstep_ankle_ys(base=2)
            + _footstep_ankle_ys(base=4)
            + _footstep_ankle_ys(base=6)
        )
        possessor_ids = [1] * len(ankle_ys)
        pose_tracks = [{1: _pose_with_ankle_y(y, conf=0.1)} for y in ankle_ys]

        violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(violations, [])

    def test_violation_span_shorter_than_min_frame_span_suppressed(self):
        # Reuse the exact fixture from test_four_steps_no_dribble_flags_traveling
        # (which, at the real MIN_VIOLATION_FRAME_SPAN=3, DOES flag a
        # violation), but patch MIN_VIOLATION_FRAME_SPAN to a value larger
        # than that fixture's actual span -- proving the span guard itself
        # suppresses an otherwise-exceeded window, isolated from step counting.
        import unittest.mock as mock

        ankle_ys = (
            _footstep_ankle_ys(base=0)
            + _footstep_ankle_ys(base=2)
            + _footstep_ankle_ys(base=4)
            + _footstep_ankle_ys(base=6)
        )
        possessor_ids = [1] * len(ankle_ys)
        pose_tracks = [{1: _pose_with_ankle_y(y)} for y in ankle_ys]

        with mock.patch(
            "personal.basketball_analysis.rule_violation_detector.traveling_detector.MIN_VIOLATION_FRAME_SPAN",
            len(ankle_ys) + 1,  # larger than this fixture's actual span -- must suppress
        ):
            violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(violations, [])

    def test_no_possession_returns_empty_list(self):
        possessor_ids = [-1, -1, -1, -1]
        pose_tracks = [{} for _ in possessor_ids]

        violations = self._run(possessor_ids, pose_tracks)

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
