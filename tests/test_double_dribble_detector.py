import unittest
from rule_violation_detector.double_dribble_detector import DoubleDribbleDetector


WRIST_X, WRIST_Y = 0.0, 0.0


def _ball_track(cx, cy, half_size=5):
    return {1: {"bbox": [cx - half_size, cy - half_size, cx + half_size, cy + half_size], "source": "detection"}}


def _pose_both_wrists_at(x, y, conf=1.0):
    keypoints = [[0.0, 0.0, 1.0]] * 17
    keypoints[9] = [x, y, conf]   # left wrist
    keypoints[10] = [x, y, conf]  # right wrist -- both at the same spot so
    # both-hand pickup checks and nearer-wrist dribble checks agree.
    return {"keypoints": keypoints, "bbox": [0, 0, 50, 100]}


def _dribble_cycle_ball_positions():
    """One complete near->far->near dribble cycle around (WRIST_X, WRIST_Y), as a
    list of (cx, cy) ball-center positions, matching the shape verified by
    test_dribble_event_detector.test_single_clean_bounce_detected."""
    return [(d, 0) for d in (5, 20, 60, 90, 60, 20, 5, 5, 5, 5)]


def _held_ball_positions(count):
    """`count` frames of the ball parked at the wrist -- within
    PICKUP_PROXIMITY_PX=35, well under it at distance 0."""
    return [(WRIST_X, WRIST_Y)] * count


class TestDoubleDribbleDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DoubleDribbleDetector()

    def _run(self, ball_positions, possessor_id=1):
        possessor_ids = [possessor_id] * len(ball_positions)
        ball_tracks = [_ball_track(cx, cy) for cx, cy in ball_positions]
        pose_tracks = [{possessor_id: _pose_both_wrists_at(WRIST_X, WRIST_Y)} for _ in ball_positions]
        return self.detector.detect(ball_tracks, possessor_ids, pose_tracks)

    def test_continuous_possession_one_dribble_no_pickup_no_violation(self):
        # A single dribble cycle, then the ball stays near the hand but never
        # forms a deliberate two-hand PICKUP_MIN_CONSECUTIVE_FRAMES hold
        # (it's just naturally near after the bounce) -- no violation.
        positions = _dribble_cycle_ball_positions()
        violations = self._run(positions)
        self.assertEqual(violations, [])

    def test_dribble_then_pickup_then_pass_no_violation(self):
        # Legal: player 1 dribbles (frames 0-9), picks up with two hands long
        # enough to reach HELD (frames 10-15), then possession passes to
        # player 2 (frames 16-25) -- no violation, since player 1 never
        # dribbles again after reaching HELD; the possession change simply
        # ends their streak.
        dribble_positions = _dribble_cycle_ball_positions()  # frames 0-9, player 1
        held_positions = _held_ball_positions(6)               # frames 10-15, player 1, reaches HELD
        pass_positions = _dribble_cycle_ball_positions()      # frames 16-25, player 2's own ball-handling

        possessor_ids = [1] * len(dribble_positions) + [1] * len(held_positions) + [2] * len(pass_positions)
        positions = dribble_positions + held_positions + pass_positions

        ball_tracks = [_ball_track(cx, cy) for cx, cy in positions]
        pose_tracks = [
            {1: _pose_both_wrists_at(WRIST_X, WRIST_Y), 2: _pose_both_wrists_at(WRIST_X, WRIST_Y)}
            for _ in positions
        ]

        violations = self.detector.detect(ball_tracks, possessor_ids, pose_tracks)
        self.assertEqual(violations, [])

    def test_dribble_two_hand_pickup_dribble_again_flags_double_dribble(self):
        # The canonical violation: dribble cycle, two-hand pickup held long
        # enough to count, then a SECOND dribble cycle for the same possessor
        # with no possession change in between.
        dribble_1 = _dribble_cycle_ball_positions()          # frames 0-9, ends near (5,0)
        held = _held_ball_positions(6)                        # frames 10-15, held at (0,0) -- >= PICKUP_MIN_CONSECUTIVE_FRAMES=5
        dribble_2 = _dribble_cycle_ball_positions()          # frames 16-25, a second full cycle
        positions = dribble_1 + held + dribble_2

        violations = self._run(positions)

        self.assertEqual(len(violations), 1)
        violation = violations[0]
        self.assertEqual(violation["violation_type"], "double_dribble")
        self.assertEqual(violation["player_id"], 1)
        self.assertEqual(violation["confidence"], "heuristic")
        self.assertLess(violation["start_frame"], violation["end_frame"])

    def test_pickup_below_min_consecutive_frames_not_counted_as_held(self):
        # Two-hand proximity only holds for 2 frames (< PICKUP_MIN_CONSECUTIVE_FRAMES=5)
        # before a second dribble cycle -- should never reach HELD, so no violation.
        # A single "ball moves away" frame is inserted right after the first
        # dribble's near tail to break its own proximity streak, so only the
        # intentional 2-frame brief_hold contributes to the consecutive count.
        # The excursion (36px) clears PICKUP_PROXIMITY_PX=35 but stays well
        # under DRIBBLE_MIN_AMPLITUDE_PX=40, so it can't itself register as a
        # dribble bounce and contaminate dribble_frames_by_player.
        dribble_1 = _dribble_cycle_ball_positions()
        interruption = [(36, 0)]  # just past PICKUP_PROXIMITY_PX, well under DRIBBLE_MIN_AMPLITUDE_PX
        brief_hold = _held_ball_positions(2)
        dribble_2 = _dribble_cycle_ball_positions()
        positions = dribble_1 + interruption + brief_hold + dribble_2

        violations = self._run(positions)

        self.assertEqual(violations, [])

    def test_possession_change_between_pickup_and_next_dribble_resets_no_violation(self):
        # Player 1 dribbles, picks up (long enough to count), then LOSES
        # possession (steal/rebound) before dribbling again -- not a violation,
        # since a legal turnover/steal resets to NOT_POSSESSING.
        dribble_1 = _dribble_cycle_ball_positions()   # 10 frames, player 1
        held = _held_ball_positions(6)                 # 6 frames, player 1, now HELD
        gap = [(WRIST_X, WRIST_Y)] * 3                  # 3 frames with nobody possessing
        dribble_2 = _dribble_cycle_ball_positions()   # 10 frames, player 2 dribbles

        possessor_ids = [1] * 10 + [1] * 6 + [-1] * 3 + [2] * 10
        positions = dribble_1 + held + gap + dribble_2

        ball_tracks = [_ball_track(cx, cy) for cx, cy in positions]
        pose_tracks = [
            {1: _pose_both_wrists_at(WRIST_X, WRIST_Y), 2: _pose_both_wrists_at(WRIST_X, WRIST_Y)}
            for _ in positions
        ]

        violations = self.detector.detect(ball_tracks, possessor_ids, pose_tracks)
        self.assertEqual(violations, [])

    def test_two_separate_sequences_same_player_flags_twice(self):
        # After one flagged+reset violation, a fully independent later
        # dribble->pickup->dribble sequence for the SAME player should flag
        # again, not be suppressed forever by the earlier reset. Note that
        # after a violation resets state to NOT_POSSESSING, a fresh dribble
        # is required to re-enter DRIBBLING before the next pickup+dribble
        # can be evaluated at all -- so each "sequence" below is a complete
        # dribble -> hold -> dribble unit on its own.
        first_sequence = (
            _dribble_cycle_ball_positions()
            + _held_ball_positions(6)
            + _dribble_cycle_ball_positions()   # completes violation #1, resets to NOT_POSSESSING
        )
        second_sequence = (
            _dribble_cycle_ball_positions()
            + _held_ball_positions(6)
            + _dribble_cycle_ball_positions()   # completes violation #2
        )
        sequence = first_sequence + second_sequence

        violations = self._run(sequence)

        self.assertEqual(len(violations), 2)
        for violation in violations:
            self.assertEqual(violation["player_id"], 1)
            self.assertEqual(violation["violation_type"], "double_dribble")
        # The second violation must start strictly after the first one ends.
        self.assertLess(violations[0]["end_frame"], violations[1]["start_frame"])

    def test_no_possession_returns_empty_list(self):
        positions = _dribble_cycle_ball_positions()
        possessor_ids = [-1] * len(positions)
        ball_tracks = [_ball_track(cx, cy) for cx, cy in positions]
        pose_tracks = [{} for _ in positions]

        violations = self.detector.detect(ball_tracks, possessor_ids, pose_tracks)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
