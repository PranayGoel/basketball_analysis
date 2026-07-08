import unittest
from personal.basketball_analysis.utils.bbox_utils import get_center_of_bbox, get_bbox_width, measure_distance, measure_xy_distance, get_foot_position


class TestBboxUtils(unittest.TestCase):
    def test_get_center_of_bbox(self):
        self.assertEqual(get_center_of_bbox((0, 0, 10, 20)), (5, 10))

    def test_get_bbox_width(self):
        self.assertEqual(get_bbox_width((10, 0, 30, 50)), 20)

    def test_measure_distance(self):
        self.assertAlmostEqual(measure_distance((0, 0), (3, 4)), 5.0)

    def test_measure_xy_distance(self):
        self.assertEqual(measure_xy_distance((10, 20), (3, 5)), (7, 15))

    def test_get_foot_position(self):
        self.assertEqual(get_foot_position((0, 0, 10, 20)), (5, 20))


if __name__ == "__main__":
    unittest.main()
