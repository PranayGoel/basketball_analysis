import unittest
from pass_and_interception_detector.pass_and_interception_detector import PassAndInterceptionDetector


class TestPassAndInterceptionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PassAndInterceptionDetector()

    def test_detect_pass_same_team(self):
        # frame0: player 1 (team1) has ball; frame1: player 2 (team1) has ball -> a pass.
        ball_acquisition = [1, 2]
        player_assignment = [{1: 1, 2: 1}, {1: 1, 2: 1}]
        passes = self.detector.detect_passes(ball_acquisition, player_assignment)
        self.assertEqual(passes, [-1, 1])

    def test_detect_interception_different_team(self):
        # frame0: player 1 (team1) has ball; frame1: player 2 (team2) has ball -> interception by team2.
        ball_acquisition = [1, 2]
        player_assignment = [{1: 1, 2: 2}, {1: 1, 2: 2}]
        interceptions = self.detector.detect_interceptions(ball_acquisition, player_assignment)
        self.assertEqual(interceptions, [-1, 2])

    def test_no_event_when_possession_unchanged(self):
        ball_acquisition = [1, 1, 1]
        player_assignment = [{1: 1}, {1: 1}, {1: 1}]
        self.assertEqual(self.detector.detect_passes(ball_acquisition, player_assignment), [-1, -1, -1])
        self.assertEqual(self.detector.detect_interceptions(ball_acquisition, player_assignment), [-1, -1, -1])

    def test_no_event_during_uncertain_possession_gap(self):
        ball_acquisition = [1, -1, 2]
        player_assignment = [{1: 1, 2: 1}, {1: 1, 2: 1}, {1: 1, 2: 1}]
        passes = self.detector.detect_passes(ball_acquisition, player_assignment)
        # possession lapses to -1 for a frame, then resumes with the same prev_holder tracked --
        # still same team, so this is a pass at the frame it's regained, not a broken sequence.
        self.assertEqual(passes[1], -1)


if __name__ == "__main__":
    unittest.main()
