"""
Job route tests. The poll endpoint (GET /api/jobs/{id}) is tested thoroughly
against directly-inserted Job rows. The SSE endpoint is tested only for the
"already terminal" fast path (no Redis needed -- see get_job route's
short-circuit for jobs already done/failed by connect time) and for basic
Content-Type/no-crash behavior; a full live-stream test would need a running
Redis and is skipped with a clear reason per the spec's guidance that this is
acceptable when testing a real streaming response is too fragile without
infrastructure.
"""

import socket
from datetime import datetime, timezone

from app.db.models import Job, Video
from tests.test_utils import BackendTestCase


def _redis_reachable(host="localhost", port=6379, timeout=0.2):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class TestGetJobStatus(BackendTestCase):
    def _insert_video_and_job(self, job_id="j1", video_id="v1", **job_kwargs):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status="processing",
                    has_violations=False,
                )
            )
            db.add(
                Job(
                    id=job_id,
                    video_id=video_id,
                    status=job_kwargs.pop("status", "processing"),
                    created_at=datetime.now(timezone.utc),
                    **job_kwargs,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_returns_404_for_missing_job(self):
        response = self.client.get("/api/jobs/does-not-exist")
        self.assertEqual(response.status_code, 404)

    def test_returns_job_status_fields(self):
        self._insert_video_and_job(
            status="processing", stage="player_tracking", stage_index=1, total_stages=13
        )
        response = self.client.get("/api/jobs/j1")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["id"], "j1")
        self.assertEqual(body["video_id"], "v1")
        self.assertEqual(body["status"], "processing")
        self.assertEqual(body["stage"], "player_tracking")
        self.assertEqual(body["stage_index"], 1)
        self.assertEqual(body["total_stages"], 13)

    def test_returns_done_status(self):
        self._insert_video_and_job(status="done", stage="report", stage_index=12, total_stages=13)
        response = self.client.get("/api/jobs/j1")
        body = response.json()
        self.assertEqual(body["status"], "done")

    def test_returns_failed_status_with_error_message(self):
        self._insert_video_and_job(status="failed", error_message="pipeline blew up")
        response = self.client.get("/api/jobs/j1")
        body = response.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error_message"], "pipeline blew up")


class TestJobStream(BackendTestCase):
    def _insert_video_and_job(self, job_id="j1", video_id="v1", **job_kwargs):
        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status="done",
                    has_violations=False,
                )
            )
            db.add(
                Job(
                    id=job_id,
                    video_id=video_id,
                    status=job_kwargs.pop("status", "done"),
                    created_at=datetime.now(timezone.utc),
                    **job_kwargs,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_returns_404_for_missing_job(self):
        response = self.client.get("/api/jobs/does-not-exist/stream")
        self.assertEqual(response.status_code, 404)

    def test_already_done_job_returns_immediate_terminal_event(self):
        # This path requires no Redis at all -- get_job's short-circuit for
        # jobs already terminal by connect time returns one SSE event and
        # closes, matching a client reconnecting/refreshing after completion.
        #
        # The "event: done" line matters, not just the JSON body: the
        # frontend's EventSource only routes a frame to its "done" listener
        # (useJobProgress.ts) when the frame actually sets that named event
        # type -- an unnamed `data:`-only frame defaults to type "message",
        # which nothing listens for, and silently drops the terminal signal.
        self._insert_video_and_job(status="done", stage="report")
        response = self.client.get("/api/jobs/j1/stream")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertIn("event: done", response.text)
        self.assertIn('"terminal": true', response.text)
        self.assertIn('"status": "done"', response.text)

    def test_already_failed_job_returns_immediate_error_event(self):
        # status must be "failed" (matching Job.status and the frontend's
        # JobProgressState union) and the field must be "error_message" (what
        # useJobProgress.ts's toProgressState() actually reads) -- not the
        # "error"/"error" status+field pairing this endpoint used to send,
        # which the frontend silently ignored.
        self._insert_video_and_job(status="failed", error_message="boom")
        response = self.client.get("/api/jobs/j1/stream")
        self.assertEqual(response.status_code, 200)
        self.assertIn("event: done", response.text)
        self.assertIn('"status": "failed"', response.text)
        self.assertIn('"error_message": "boom"', response.text)

    def test_live_stream_relays_a_published_progress_event(self):
        """
        Exercises the REAL Redis pub/sub path (not the already-terminal
        short-circuit): publish one terminal progress event to the job's
        channel via a plain sync redis client (mirroring exactly what
        worker/tasks.py's _publish_progress does), then connect to the SSE
        endpoint and confirm it relays that event and closes.

        Publishing *before* connecting -- rather than racing a publish against
        a live subscribe from a second thread -- keeps this deterministic:
        redis-py's pubsub delivers messages published after subscribe, so a
        message published first and then subscribed-to would normally be
        lost, EXCEPT the endpoint's event_generator() polls in a loop with
        get_message(timeout=1.0) which will simply see nothing and keep
        polling if subscribed after the publish -- so instead this drives the
        publish from a short-lived background thread that waits just long
        enough for the subscription to be live before firing, which is the
        realistic ordering (worker publishes only after a client is already
        watching, in the real system, but is not guaranteed here). A bounded
        response-read prevents this test from ever hanging if that race is
        lost: this test is intentionally best-effort against real infra and
        skips outright if Redis isn't reachable, per the spec's guidance that
        a fragile streaming-infra test may be kept minimal/skip-if-unavailable.
        """
        if not _redis_reachable():
            self.skipTest(
                "Redis not reachable on localhost:6379 -- the live (non-terminal) "
                "SSE stream path needs a real Redis pub/sub channel and is skipped "
                "here. Start Redis (redis-server --daemonize yes) to exercise it."
            )

        import json
        import threading
        import time

        import redis as sync_redis

        self._insert_video_and_job(status="processing", stage="read_video")

        def publish_after_delay():
            time.sleep(0.3)
            client = sync_redis.from_url("redis://localhost:6379")
            try:
                client.publish(
                    "job:j1:progress",
                    json.dumps({"stage": "report", "status": "done", "terminal": True}),
                )
            finally:
                client.close()

        publisher = threading.Thread(target=publish_after_delay, daemon=True)
        publisher.start()

        received_lines = []
        try:
            with self.client.stream("GET", "/api/jobs/j1/stream") as response:
                self.assertEqual(response.status_code, 200)
                deadline = time.time() + 5.0
                for line in response.iter_lines():
                    if line:
                        received_lines.append(line)
                    if any("terminal" in ln for ln in received_lines):
                        break
                    if time.time() > deadline:
                        break
        finally:
            publisher.join(timeout=2.0)

        self.assertTrue(
            any("terminal" in ln for ln in received_lines),
            f"Expected a terminal SSE event, got: {received_lines}",
        )
        # The published payload has no "event:" line of its own -- the route
        # is responsible for adding one based on the payload's "terminal"
        # flag. Without "event: done", the frontend's EventSource never
        # dispatches to its "done" listener (see useJobProgress.ts) and the
        # UI falls back to treating the eventual connection close as a
        # generic "Lost connection" failure, even on real success.
        self.assertIn(
            "event: done",
            received_lines,
            f"Expected a named 'event: done' SSE line, got: {received_lines}",
        )
