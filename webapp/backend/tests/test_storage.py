"""
storage.py path-management tests, fully isolated in a temp dir -- no real
video needed, this only tests path construction, upload-copying, and delete
semantics.
"""

import io
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from app.services import storage


class TestGetPaths(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.settings_patch = patch.object(storage, "settings")
        mock_settings = self.settings_patch.start()
        mock_settings.uploads_dir = os.path.join(self.tmp_dir, "uploads")
        mock_settings.outputs_dir = os.path.join(self.tmp_dir, "outputs")
        mock_settings.reports_dir = os.path.join(self.tmp_dir, "reports")

    def tearDown(self):
        self.settings_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_paths_is_deterministic_for_same_video_id(self):
        paths_a = storage.get_paths("abc123")
        paths_b = storage.get_paths("abc123")
        self.assertEqual(paths_a.upload_path, paths_b.upload_path)
        self.assertEqual(paths_a.output_path, paths_b.output_path)
        self.assertEqual(paths_a.report_path, paths_b.report_path)

    def test_get_paths_uses_given_extension(self):
        paths = storage.get_paths("abc123", upload_ext=".mov")
        self.assertTrue(paths.upload_path.endswith("abc123.mov"))

    def test_get_paths_creates_parent_directories(self):
        storage.get_paths("abc123")
        self.assertTrue(os.path.isdir(os.path.join(self.tmp_dir, "uploads")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp_dir, "outputs")))
        self.assertTrue(os.path.isdir(os.path.join(self.tmp_dir, "reports")))

    def test_output_and_report_paths_use_expected_extensions(self):
        paths = storage.get_paths("abc123")
        self.assertTrue(paths.output_path.endswith("abc123.mp4"))
        self.assertTrue(paths.report_path.endswith("abc123.json"))


class TestSaveUpload(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.settings_patch = patch.object(storage, "settings")
        mock_settings = self.settings_patch.start()
        mock_settings.uploads_dir = os.path.join(self.tmp_dir, "uploads")
        mock_settings.outputs_dir = os.path.join(self.tmp_dir, "outputs")
        mock_settings.reports_dir = os.path.join(self.tmp_dir, "reports")

    def tearDown(self):
        self.settings_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_upload_writes_file_contents(self):
        fake_file = io.BytesIO(b"pretend this is video bytes")
        paths = storage.save_upload("vid1", "clip.mp4", fake_file)

        self.assertTrue(os.path.isfile(paths.upload_path))
        with open(paths.upload_path, "rb") as f:
            self.assertEqual(f.read(), b"pretend this is video bytes")

    def test_save_upload_derives_extension_from_filename(self):
        fake_file = io.BytesIO(b"data")
        paths = storage.save_upload("vid1", "clip.mov", fake_file)
        self.assertTrue(paths.upload_path.endswith("vid1.mov"))

    def test_save_upload_defaults_extension_when_filename_has_none(self):
        fake_file = io.BytesIO(b"data")
        paths = storage.save_upload("vid1", "clip", fake_file)
        self.assertTrue(paths.upload_path.endswith("vid1.mp4"))


class TestDeleteVideoFiles(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.settings_patch = patch.object(storage, "settings")
        mock_settings = self.settings_patch.start()
        mock_settings.uploads_dir = os.path.join(self.tmp_dir, "uploads")
        mock_settings.outputs_dir = os.path.join(self.tmp_dir, "outputs")
        mock_settings.reports_dir = os.path.join(self.tmp_dir, "reports")

    def tearDown(self):
        self.settings_patch.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_deletes_all_existing_files_for_video_id(self):
        fake_file = io.BytesIO(b"data")
        paths = storage.save_upload("vid1", "clip.mp4", fake_file)
        with open(paths.output_path, "wb") as f:
            f.write(b"annotated video")
        with open(paths.report_path, "w") as f:
            f.write("{}")

        storage.delete_video_files("vid1")

        self.assertFalse(os.path.isfile(paths.upload_path))
        self.assertFalse(os.path.isfile(paths.output_path))
        self.assertFalse(os.path.isfile(paths.report_path))

    def test_delete_is_safe_when_no_files_exist(self):
        # A video that failed before its output video/report were ever
        # written should still delete cleanly -- missing files are not an
        # error.
        try:
            storage.delete_video_files("never-existed")
        except Exception as exc:  # noqa: BLE001
            self.fail(f"delete_video_files raised unexpectedly: {exc}")
