import unittest
from personal.basketball_analysis.rule_violation_detector.dribble_event_detector import detect_dribble_events


def _ball_track(cx, cy, half_size=5):
    return {1: {"bbox": [cx - half_size, cy - half_size, cx + half_size, cy + half_size], "source": "detection"}}


def _pose_with_wrist(wrist_x, wrist_y, wrist_conf=1.0):
    keypoints = [[0.0, 0.0, 1.0]] * 17
    keypoints[9] = [wrist_x, wrist_y, wrist_conf]   # left wrist
    keypoints[10] = [1000.0, 1000.0, 0.0]           # right wrist, far away + low conf, ignored
    return {"keypoints": keypoints, "bbox": [0, 0, 50, 100]}


class TestDetectDribbleEvents(unittest.TestCase):
    def test_single_clean_bounce_detected(self):
        # Wrist fixed at (0, 0). Ball goes near -> far -> near over ~10 frames.
        # distance pattern: 5, 20, 60, 80, 60, 20, 5, 5, 5, 5 (amplitude swing > 40px)
        distances = [5, 20, 60, 90, 60, 20, 5, 5, 5, 5]
        possessor_ids = [1] * len(distances)
        ball_tracks = [_ball_track(d, 0) for d in distances]
        pose_tracks = [{1: _pose_with_wrist(0, 0)} for _ in distances]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["player_id"], 1)

    def test_amplitude_below_threshold_not_counted(self):
        # Swings only 20px (< DRIBBLE_MIN_AMPLITUDE_PX=40) -- should not register.
        distances = [5, 10, 20, 25, 20, 10, 5, 5, 5, 5]
        possessor_ids = [1] * len(distances)
        ball_tracks = [_ball_track(d, 0) for d in distances]
        pose_tracks = [{1: _pose_with_wrist(0, 0)} for _ in distances]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(events, [])

    def test_cycle_too_slow_not_counted(self):
        # Amplitude is fine and the cycle DOES complete (rises then returns),
        # but it spans far more than DRIBBLE_MAX_CYCLE_FRAMES=45 frames: the
        # peak is confirmed early (frame 2, once the rise clears the
        # amplitude threshold), held for 48 more frames, then the drop at
        # frame 50 transitions into the return-tracking phase, and frame 51
        # confirms the completed-but-overlong cycle -- exercising the length
        # rejection itself, not just an unfinished cycle.
        distances = [5, 20, 90] + [90] * 47 + [45, 5]
        possessor_ids = [1] * len(distances)
        ball_tracks = [_ball_track(d, 0) for d in distances]
        pose_tracks = [{1: _pose_with_wrist(0, 0)} for _ in distances]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(events, [])

    def test_cycle_too_fast_not_counted(self):
        # Amplitude is fine but the cycle completes in 2 frames
        # (< DRIBBLE_MIN_CYCLE_FRAMES=4) -- rejected as noise.
        distances = [5, 90, 5]
        possessor_ids = [1] * len(distances)
        ball_tracks = [_ball_track(d, 0) for d in distances]
        pose_tracks = [{1: _pose_with_wrist(0, 0)} for _ in distances]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(events, [])

    def test_low_confidence_wrist_ignored(self):
        # Distance pattern looks exactly like a valid dribble, but the wrist
        # keypoint confidence is below WRIST_MIN_CONF=0.3 throughout, and the
        # other wrist is also low-confidence -- so no trustworthy distance
        # signal exists at all, and zero events should fire.
        distances = [5, 20, 60, 90, 60, 20, 5, 5, 5, 5]
        possessor_ids = [1] * len(distances)
        ball_tracks = [_ball_track(d, 0) for d in distances]
        pose_tracks = [{1: _pose_with_wrist(0, 0, wrist_conf=0.1)} for _ in distances]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(events, [])

    def test_possessor_change_mid_cycle_does_not_falsely_attribute(self):
        # Player 1 starts a dribble cycle (near -> far), then possession
        # switches to player 2 mid-cycle. Player 2's own distance pattern
        # should not be contaminated by player 1's partial cycle state, and
        # player 1's incomplete cycle should never fire since it never returns.
        possessor_ids = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        ball_tracks = [
            _ball_track(5, 0),   # p1 near
            _ball_track(60, 0),  # p1 far (cycle started, never completes)
            _ball_track(90, 0),  # p1 far
            _ball_track(5, 0),   # p2 near (new streak, fresh tracker)
            _ball_track(20, 0),
            _ball_track(60, 0),
            _ball_track(90, 0),  # p2 far
            _ball_track(60, 0),
            _ball_track(20, 0),
            _ball_track(5, 0),   # p2 near again -> completes cycle for p2
        ]
        pose_tracks = [
            {1: _pose_with_wrist(0, 0)},
            {1: _pose_with_wrist(0, 0)},
            {1: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
            {2: _pose_with_wrist(0, 0)},
        ]

        events = detect_dribble_events(ball_tracks, possessor_ids, pose_tracks)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["player_id"], 2)


if __name__ == "__main__":
    unittest.main()
