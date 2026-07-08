import unittest
import numpy as np

from personal.basketball_analysis.drawers.pose_drawer import PoseDrawer, MIN_KEYPOINT_CONF_TO_DRAW
from personal.basketball_analysis.pose_estimator.pose_estimator import NUM_COCO_KEYPOINTS


def make_keypoints(conf):
    """17 COCO keypoints spread across a small frame, all at the given confidence."""
    return [[float(i), float(i), conf] for i in range(NUM_COCO_KEYPOINTS)]


class TestPoseDrawer(unittest.TestCase):
    def setUp(self):
        self.drawer = PoseDrawer()
        self.frame = np.zeros((50, 50, 3), dtype=np.uint8)

    def test_draw_only_connects_high_confidence_keypoint_pairs(self):
        # Mix of high-confidence (above threshold) and low-confidence
        # (below threshold) keypoints in the same player's pose.
        high_conf = MIN_KEYPOINT_CONF_TO_DRAW + 0.5
        low_conf = MIN_KEYPOINT_CONF_TO_DRAW - 0.1

        keypoints = make_keypoints(high_conf)
        # Drop confidence for a few keypoints below the draw threshold.
        keypoints[0][2] = low_conf   # nose
        keypoints[7][2] = low_conf   # left elbow
        keypoints[16][2] = low_conf  # right ankle

        pose_tracks = [
            {1: {"keypoints": keypoints, "bbox": [0, 0, 10, 10], "crop_bbox": [0, 0, 10, 10]}}
        ]

        result = self.drawer.draw([self.frame], pose_tracks)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, self.frame.shape)

    def test_draw_handles_empty_pose_dict_without_crashing(self):
        # A frame where no players were detected that frame.
        pose_tracks = [{}]

        result = self.drawer.draw([self.frame], pose_tracks)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, self.frame.shape)

    def test_draw_handles_all_low_confidence_keypoints_without_crashing(self):
        low_conf = MIN_KEYPOINT_CONF_TO_DRAW - 0.05
        keypoints = make_keypoints(low_conf)

        pose_tracks = [
            {1: {"keypoints": keypoints, "bbox": [0, 0, 10, 10], "crop_bbox": [0, 0, 10, 10]}}
        ]

        result = self.drawer.draw([self.frame], pose_tracks)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, self.frame.shape)

    def test_draw_returns_same_number_of_frames_as_input(self):
        high_conf = MIN_KEYPOINT_CONF_TO_DRAW + 0.5
        keypoints = make_keypoints(high_conf)

        frames = [self.frame.copy() for _ in range(3)]
        pose_tracks = [
            {1: {"keypoints": keypoints, "bbox": [0, 0, 10, 10], "crop_bbox": [0, 0, 10, 10]}},
            {},
            {2: {"keypoints": keypoints, "bbox": [5, 5, 15, 15], "crop_bbox": [5, 5, 15, 15]}},
        ]

        result = self.drawer.draw(frames, pose_tracks)

        self.assertEqual(len(result), len(frames))


if __name__ == "__main__":
    unittest.main()
