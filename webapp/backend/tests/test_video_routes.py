"""
Video route tests. Upload mocks the arq enqueue (services.ingestion.enqueue_pipeline_job)
so no real Redis/worker is needed. Library filter/sort tests insert Video rows
directly via the test DB session, bypassing the upload flow entirely.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

from personal.basketball_analysis.webapp.backend.app.db.models import Video
from personal.basketball_analysis.webapp.backend.tests.test_utils import BackendTestCase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLE_VIDEO_PATH = os.path.join(REPO_ROOT, "input_videos", "video_1.mp4")


class TestUploadVideo(BackendTestCase):
    @patch("personal.basketball_analysis.webapp.backend.app.services.ingestion.enqueue_pipeline_job", new_callable=AsyncMock)
    def test_upload_creates_video_and_job_rows(self, mock_enqueue):
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            response = self.client.post(
                "/api/videos",
                files={"file": ("video_1.mp4", f, "video/mp4")},
            )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertIn("video_id", body)
        self.assertIn("job_id", body)
        self.assertEqual(body["status"], "queued")

        db = self.db_session()
        try:
            video = db.get(Video, body["video_id"])
            self.assertIsNotNone(video)
            self.assertEqual(video.filename, "video_1.mp4")
            self.assertEqual(video.status, "queued")
            self.assertTrue(os.path.isfile(video.upload_path))
        finally:
            db.close()

        mock_enqueue.assert_awaited_once_with(body["job_id"])

    @patch("personal.basketball_analysis.webapp.backend.app.services.ingestion.enqueue_pipeline_job", new_callable=AsyncMock)
    def test_upload_does_not_require_real_redis(self, mock_enqueue):
        # If this test passes without a running Redis/arq worker, the mock is
        # doing its job -- the whole point of patching enqueue_pipeline_job.
        with open(SAMPLE_VIDEO_PATH, "rb") as f:
            response = self.client.post("/api/videos", files={"file": ("v.mp4", f, "video/mp4")})
        self.assertEqual(response.status_code, 201)


class TestListVideos(BackendTestCase):
    def _insert_video(self, video_id, status="done", possession_a=50.0, total_passes=0, max_speed=0.0, days_ago=0):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename=f"{video_id}.mp4",
                    uploaded_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status=status,
                    team_a_possession_pct=possession_a,
                    team_b_possession_pct=100 - possession_a if possession_a is not None else None,
                    total_passes=total_passes,
                    max_player_speed_kmh=max_speed,
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_filters_by_status(self):
        self._insert_video("v1", status="done")
        self._insert_video("v2", status="failed")

        response = self.client.get("/api/videos", params={"status": "done"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["items"][0]["id"], "v1")

    def test_filters_by_min_possession_split(self):
        # v1 is a near-even split (|52-50|=2), v2 is a blowout (|90-50|=40).
        self._insert_video("v1", possession_a=52.0)
        self._insert_video("v2", possession_a=90.0)

        response = self.client.get("/api/videos", params={"min_possession_split": 5})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        ids = [item["id"] for item in body["items"]]
        self.assertIn("v1", ids)
        self.assertNotIn("v2", ids)

    def test_sorts_by_total_passes_desc(self):
        self._insert_video("v1", total_passes=3)
        self._insert_video("v2", total_passes=10)
        self._insert_video("v3", total_passes=1)

        response = self.client.get("/api/videos", params={"sort_by": "total_passes", "sort_order": "desc"})
        body = response.json()
        ids = [item["id"] for item in body["items"]]
        self.assertEqual(ids, ["v2", "v1", "v3"])

    def test_sorts_by_max_player_speed_kmh_asc(self):
        self._insert_video("v1", max_speed=20.0)
        self._insert_video("v2", max_speed=5.0)

        response = self.client.get(
            "/api/videos", params={"sort_by": "max_player_speed_kmh", "sort_order": "asc"}
        )
        body = response.json()
        ids = [item["id"] for item in body["items"]]
        self.assertEqual(ids, ["v2", "v1"])

    def test_pagination(self):
        for i in range(5):
            self._insert_video(f"v{i}")

        response = self.client.get("/api/videos", params={"page": 1, "page_size": 2})
        body = response.json()
        self.assertEqual(len(body["items"]), 2)
        self.assertEqual(body["total"], 5)
        self.assertEqual(body["page"], 1)
        self.assertEqual(body["page_size"], 2)


class TestGetVideoDetail(BackendTestCase):
    def test_returns_404_for_missing_id(self):
        response = self.client.get("/api/videos/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_returns_full_detail_for_existing_video(self):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id="v1",
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path="/tmp/v1.mp4",
                    status="done",
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

        response = self.client.get("/api/videos/v1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], "v1")
        self.assertEqual(body["upload_path"], "/tmp/v1.mp4")


class TestDeleteVideo(BackendTestCase):
    def test_returns_404_for_missing_id(self):
        response = self.client.delete("/api/videos/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_deletes_video_row_and_files(self):
        # BackendTestCase already patches storage's uploads_dir to
        # self.tmp_dir/uploads -- write the fixture file there directly so
        # delete_video_files() (which globs settings.uploads_dir for any
        # extension matching this video_id) finds and removes it for real.
        upload_dir = os.path.join(self.tmp_dir, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        upload_path = os.path.join(upload_dir, "v1.mp4")
        with open(upload_path, "wb") as f:
            f.write(b"fake video bytes")

        db = self.db_session()
        try:
            db.add(
                Video(
                    id="v1",
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=upload_path,
                    status="done",
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

        response = self.client.delete("/api/videos/v1")
        self.assertEqual(response.status_code, 204)

        self.assertFalse(os.path.isfile(upload_path))

        db = self.db_session()
        try:
            self.assertIsNone(db.get(Video, "v1"))
        finally:
            db.close()
