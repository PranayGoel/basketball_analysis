import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from personal.basketball_analysis.pose_estimator.pose_estimator import PoseEstimator, NUM_COCO_KEYPOINTS
from personal.basketball_analysis.utils.stubs_utils import save_stub


class FakeTensor:
    """
    A minimal stand-in for a torch tensor, replicating just the surface
    pose_estimator.py actually touches: indexing (returning nested FakeTensors
    down to a leaf value), .tolist() on a leaf row, float() conversion, len(),
    and .argmax() -- so tests never need torch/ultralytics installed.
    """

    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        value = self._data[idx]
        if isinstance(value, list):
            return FakeTensor(value)
        return value

    def __len__(self):
        return len(self._data)

    def __float__(self):
        return float(self._data)

    def tolist(self):
        return list(self._data)

    def argmax(self):
        return self._data.index(max(self._data))


def make_fake_result(detections):
    """
    Build a fake YOLO pose result for one crop.

    Args:
        detections: list of (keypoints_xy, keypoints_conf, box_conf) tuples,
            one per detected person in this crop. keypoints_xy is a list of
            17 [x, y] pairs; keypoints_conf is a list of 17 floats.

    Returns:
        MagicMock: an object with .keypoints.xy, .keypoints.conf, .boxes.conf
        matching the shapes pose_estimator.py expects.
    """
    result = MagicMock()
    result.keypoints.xy = FakeTensor([d[0] for d in detections])
    result.keypoints.conf = FakeTensor([d[1] for d in detections])
    result.boxes.conf = FakeTensor([d[2] for d in detections])
    return result


def make_keypoints(x_offset=0.0, y_offset=0.0, conf=0.9):
    """Build a simple set of 17 COCO keypoints, all at a fixed offset/confidence."""
    xy = [[x_offset + i, y_offset + i] for i in range(NUM_COCO_KEYPOINTS)]
    conf_list = [conf] * NUM_COCO_KEYPOINTS
    return xy, conf_list


class TestPadBbox(unittest.TestCase):
    @patch('pose_estimator.pose_estimator.YOLO')
    def setUp(self, mock_yolo):
        self.estimator = PoseEstimator(model_path='fake.pt')

    def test_pad_bbox_clamped_to_frame_bounds(self):
        # Bbox near a frame edge in a 100x100 frame.
        bbox = [5, 5, 50, 50]
        px1, py1, px2, py2 = self.estimator._pad_bbox(bbox, frame_w=100, frame_h=100)

        self.assertGreaterEqual(px1, 0)
        self.assertGreaterEqual(py1, 0)
        self.assertLessEqual(px2, 100)
        self.assertLessEqual(py2, 100)
        # Should still be a valid (non-degenerate) box.
        self.assertGreater(px2, px1)
        self.assertGreater(py2, py1)

    def test_pad_bbox_uses_fraction_and_floor(self):
        # A small bbox where 15% of its size is well under 10px --
        # padding should still be at least bbox_pad_min_px (10).
        bbox = [40, 40, 45, 45]  # 5x5 box; 15% of 5 = 0.75px
        px1, py1, px2, py2 = self.estimator._pad_bbox(bbox, frame_w=1000, frame_h=1000)

        pad_left = 40 - px1
        pad_top = 40 - py1
        self.assertGreaterEqual(pad_left, self.estimator.bbox_pad_min_px)
        self.assertGreaterEqual(pad_top, self.estimator.bbox_pad_min_px)


class TestDetectPoses(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((200, 200, 3), dtype=np.uint8)

    @patch('pose_estimator.pose_estimator.YOLO')
    def test_detect_poses_output_shape(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        xy, conf_list = make_keypoints(x_offset=2.0, y_offset=3.0, conf=0.9)
        mock_model.predict.return_value = [
            make_fake_result([(xy, conf_list, 0.95)])
        ]

        estimator = PoseEstimator(model_path='fake.pt')

        player_bbox = [50, 50, 90, 130]
        player_tracks = [{1: {"bbox": player_bbox}}]

        result = estimator.detect_poses([self.frame], player_tracks)

        self.assertEqual(len(result), 1)
        frame_poses = result[0]
        self.assertIn(1, frame_poses)

        player_pose = frame_poses[1]
        self.assertIn("keypoints", player_pose)
        self.assertIn("bbox", player_pose)
        self.assertIn("crop_bbox", player_pose)

        keypoints = player_pose["keypoints"]
        self.assertEqual(len(keypoints), NUM_COCO_KEYPOINTS)
        for entry in keypoints:
            self.assertEqual(len(entry), 3)  # [x, y, conf]

        # Original (unpadded) bbox passed through unchanged.
        self.assertEqual(player_pose["bbox"], player_bbox)

        # Keypoints translated from crop-local back to full-frame coordinates:
        # crop origin offset must be added back to the raw model output.
        crop_x1, crop_y1, _, _ = player_pose["crop_bbox"]
        expected_x0 = xy[0][0] + crop_x1
        expected_y0 = xy[0][1] + crop_y1
        self.assertAlmostEqual(keypoints[0][0], expected_x0)
        self.assertAlmostEqual(keypoints[0][1], expected_y0)
        self.assertAlmostEqual(keypoints[0][2], conf_list[0])

    @patch('pose_estimator.pose_estimator.YOLO')
    def test_detect_poses_skips_degenerate_crop(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model
        # predict should not even be meaningfully consulted for a degenerate
        # crop, but guard with an empty-list return in case it's called.
        mock_model.predict.return_value = []

        estimator = PoseEstimator(model_path='fake.pt')

        # Bbox fully out of frame bounds (frame is 200x200).
        out_of_bounds_bbox = [500, 500, 550, 600]
        player_tracks = [{7: {"bbox": out_of_bounds_bbox}}]

        result = estimator.detect_poses([self.frame], player_tracks)

        self.assertEqual(len(result), 1)
        self.assertNotIn(7, result[0])
        self.assertEqual(result[0], {})

    @patch('pose_estimator.pose_estimator.YOLO')
    def test_detect_poses_picks_highest_confidence_detection_when_multiple(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        low_xy, low_conf = make_keypoints(x_offset=1.0, y_offset=1.0, conf=0.5)
        high_xy, high_conf = make_keypoints(x_offset=99.0, y_offset=99.0, conf=0.99)

        # Two "person" detections in the same crop: lower box conf first,
        # higher box conf second -- the higher one should win regardless of order.
        mock_model.predict.return_value = [
            make_fake_result([
                (low_xy, low_conf, 0.4),
                (high_xy, high_conf, 0.9),
            ])
        ]

        estimator = PoseEstimator(model_path='fake.pt')
        player_bbox = [20, 20, 60, 100]
        player_tracks = [{3: {"bbox": player_bbox}}]

        result = estimator.detect_poses([self.frame], player_tracks)

        keypoints = result[0][3]["keypoints"]
        crop_x1, crop_y1, _, _ = result[0][3]["crop_bbox"]

        # The higher-confidence detection's keypoints should be used.
        self.assertAlmostEqual(keypoints[0][0], high_xy[0][0] + crop_x1)
        self.assertAlmostEqual(keypoints[0][1], high_xy[0][1] + crop_y1)
        self.assertAlmostEqual(keypoints[0][2], high_conf[0])


class TestGetObjectPosesStubCache(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @patch('pose_estimator.pose_estimator.YOLO')
    def test_get_object_poses_stub_cache_contract(self, mock_yolo_cls):
        import os

        stub_path = os.path.join(self.tmp_dir, "pose_stub.pkl")
        estimator = PoseEstimator(model_path='fake.pt')

        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
        player_tracks = [{} for _ in range(3)]

        cached_poses = [{}, {}, {}]
        save_stub(stub_path, cached_poses)

        with patch.object(estimator, 'detect_poses') as mock_detect:
            result = estimator.get_object_poses(
                frames, player_tracks, read_from_stub=True, stub_path=stub_path
            )
            mock_detect.assert_not_called()
            self.assertEqual(result, cached_poses)

        # Mismatched frame count -> cache considered stale, detect_poses called.
        mismatched_frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(5)]
        mismatched_tracks = [{} for _ in range(5)]

        fresh_poses = [{}, {}, {}, {}, {}]
        with patch.object(estimator, 'detect_poses', return_value=fresh_poses) as mock_detect:
            result = estimator.get_object_poses(
                mismatched_frames, mismatched_tracks, read_from_stub=True, stub_path=stub_path
            )
            mock_detect.assert_called_once_with(mismatched_frames, mismatched_tracks)
            self.assertEqual(result, fresh_poses)


if __name__ == "__main__":
    unittest.main()
