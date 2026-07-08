import unittest
from personal.basketball_analysis.speed_and_distance_calculator.speed_and_distance_calculator import SpeedAndDistanceCalculator


class TestSpeedAndDistanceCalculator(unittest.TestCase):
    def setUp(self):
        # 1 pixel == 1 meter for simplicity in these tests.
        self.calc = SpeedAndDistanceCalculator(
            width_in_pixels=100, height_in_pixels=100, width_in_meters=100, height_in_meters=100
        )

    def test_calculate_distance_skips_first_seen_frame(self):
        positions = [{1: [0.0, 0.0]}, {1: [3.0, 4.0]}]
        distances = self.calc.calculate_distance(positions)
        self.assertEqual(distances[0], {})
        self.assertIn(1, distances[1])

    def test_calculate_distance_value(self):
        positions = [{1: [0.0, 0.0]}, {1: [3.0, 4.0]}]
        distances = self.calc.calculate_distance(positions)
        # measure_distance((3,4),(0,0)) == 5, then * 0.4 fudge factor per the implementation
        self.assertAlmostEqual(distances[1][1], 2.0, places=5)

    def test_calculate_speed_zero_before_enough_samples(self):
        # Only 2 frames of movement -- below window_size=5 -- speed should be 0.0.
        distances = [{}, {1: 1.0}, {1: 1.0}]
        speeds = self.calc.calculate_speed(distances, fps=30)
        self.assertEqual(speeds[2][1], 0.0)

    def test_calculate_speed_positive_once_enough_samples(self):
        # 6 frames with the player present and moving -- above window_size=5.
        distances = [{}] + [{1: 1.0} for _ in range(6)]
        speeds = self.calc.calculate_speed(distances, fps=30)
        self.assertGreater(speeds[-1][1], 0.0)

    def test_calculate_speed_uses_passed_fps_not_hardcoded_default(self):
        # Same distances, two different fps values must give two different speeds --
        # regression guard for the fix that threads the real source fps through instead
        # of always assuming 30.
        distances = [{}] + [{1: 1.0} for _ in range(6)]
        speeds_30fps = self.calc.calculate_speed(distances, fps=30)
        speeds_60fps = self.calc.calculate_speed(distances, fps=60)
        self.assertNotAlmostEqual(speeds_30fps[-1][1], speeds_60fps[-1][1])


if __name__ == "__main__":
    unittest.main()
