import os
import tempfile
import unittest
from unittest import mock

from pipeline import model_resolution
from pipeline.model_resolution import (
    resolve_player_model,
    resolve_ball_model,
    resolve_court_keypoint_model,
    describe_resolution,
    COCO_FALLBACK_MODEL,
    COCO_PERSON_CLASS,
    COCO_SPORTS_BALL_CLASS,
)


class TestResolvePlayerModel(unittest.TestCase):
    def setUp(self):
        # Never let a real/missing PLAYER_DETECTOR_PATH or a leftover env var
        # from the actual shell bleed into these tests -- each test sets up
        # its own isolated view of both.
        self._env_patch = mock.patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop(model_resolution.PLAYER_OVERRIDE_ENV_VAR, None)

    def tearDown(self):
        self._env_patch.stop()

    def test_falls_back_to_coco_when_no_custom_weights_and_no_override(self):
        with mock.patch("pipeline.model_resolution.PLAYER_DETECTOR_PATH", "/nonexistent/player_detector.pt"):
            resolution = resolve_player_model()
        self.assertEqual(resolution.source, "fallback")
        self.assertEqual(resolution.weights_path, COCO_FALLBACK_MODEL)
        self.assertEqual(resolution.class_name, COCO_PERSON_CLASS)

    def test_prefers_custom_weights_when_present(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_weights:
            with mock.patch("pipeline.model_resolution.PLAYER_DETECTOR_PATH", tmp_weights.name):
                resolution = resolve_player_model()
        self.assertEqual(resolution.source, "custom")
        self.assertEqual(resolution.class_name, "Player")

    def test_env_override_wins_even_when_custom_weights_exist(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_weights:
            os.environ[model_resolution.PLAYER_OVERRIDE_ENV_VAR] = "/some/override/path.pt"
            try:
                with mock.patch("pipeline.model_resolution.PLAYER_DETECTOR_PATH", tmp_weights.name):
                    resolution = resolve_player_model()
            finally:
                os.environ.pop(model_resolution.PLAYER_OVERRIDE_ENV_VAR, None)
        self.assertEqual(resolution.source, "override")
        self.assertEqual(resolution.weights_path, "/some/override/path.pt")
        self.assertEqual(resolution.class_name, "Player")


class TestResolveBallModel(unittest.TestCase):
    def setUp(self):
        os.environ.pop(model_resolution.BALL_OVERRIDE_ENV_VAR, None)

    def test_falls_back_to_coco_sports_ball_when_no_custom_weights(self):
        with mock.patch("pipeline.model_resolution.BALL_DETECTOR_PATH", "/nonexistent/ball_detector.pt"):
            resolution = resolve_ball_model()
        self.assertEqual(resolution.source, "fallback")
        self.assertEqual(resolution.weights_path, COCO_FALLBACK_MODEL)
        self.assertEqual(resolution.class_name, COCO_SPORTS_BALL_CLASS)

    def test_prefers_custom_weights_when_present(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_weights:
            with mock.patch("pipeline.model_resolution.BALL_DETECTOR_PATH", tmp_weights.name):
                resolution = resolve_ball_model()
        self.assertEqual(resolution.source, "custom")
        self.assertEqual(resolution.class_name, "Ball")


class TestResolveCourtKeypointModel(unittest.TestCase):
    def setUp(self):
        os.environ.pop(model_resolution.COURT_KEYPOINT_OVERRIDE_ENV_VAR, None)

    def test_returns_none_when_no_custom_weights_and_no_override(self):
        # The key difference from player/ball: no fallback tier exists at all.
        with mock.patch("pipeline.model_resolution.COURT_KEYPOINT_DETECTOR_PATH", "/nonexistent/court.pt"):
            resolution = resolve_court_keypoint_model()
        self.assertIsNone(resolution)

    def test_prefers_custom_weights_when_present(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_weights:
            with mock.patch("pipeline.model_resolution.COURT_KEYPOINT_DETECTOR_PATH", tmp_weights.name):
                resolution = resolve_court_keypoint_model()
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.source, "custom")

    def test_env_override_wins_even_without_custom_weights(self):
        os.environ[model_resolution.COURT_KEYPOINT_OVERRIDE_ENV_VAR] = "/some/override/court.pt"
        try:
            with mock.patch("pipeline.model_resolution.COURT_KEYPOINT_DETECTOR_PATH", "/nonexistent/court.pt"):
                resolution = resolve_court_keypoint_model()
        finally:
            os.environ.pop(model_resolution.COURT_KEYPOINT_OVERRIDE_ENV_VAR, None)
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.source, "override")
        self.assertEqual(resolution.weights_path, "/some/override/court.pt")


class TestDescribeResolution(unittest.TestCase):
    def test_degraded_mode_message_when_none(self):
        message = describe_resolution("court_keypoints", None)
        self.assertIn("DEGRADED", message)
        self.assertIn("court_keypoints", message)

    def test_fallback_resolution_includes_accuracy_warning(self):
        from pipeline.model_resolution import ModelResolution
        fallback = ModelResolution(weights_path=COCO_FALLBACK_MODEL, class_name=COCO_SPORTS_BALL_CLASS, source="fallback")
        message = describe_resolution("ball", fallback)
        self.assertIn("fallback", message)
        self.assertIn("materially lower", message)

    def test_custom_resolution_has_no_accuracy_warning(self):
        from pipeline.model_resolution import ModelResolution
        custom = ModelResolution(weights_path="models/ball_detector_model.pt", class_name="Ball", source="custom")
        message = describe_resolution("ball", custom)
        self.assertNotIn("materially lower", message)


if __name__ == "__main__":
    unittest.main()
