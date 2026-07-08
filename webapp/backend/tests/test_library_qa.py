"""
library_qa tests: a fake/duck-typed LLM client (same pattern as the repo
root's tests/fakes.py, reused directly here since it's a stdlib-only shape
with zero dependency on which report/DB layer it's driving) with scripted
tool-call responses, verifying query_library() dispatches to the right
SQL-backed function and that find_videos_with_violations returns [] (not an
error) against an empty Violation table.
"""

import importlib.util
import json
import os
from datetime import datetime, timezone

from personal.basketball_analysis.webapp.backend.app.db.models import Player, Video
from personal.basketball_analysis.webapp.backend.app.llm.library_qa import (
    compare_videos,
    find_closest_possession_split,
    find_videos_by_min_distance,
    find_videos_with_violations,
    query_library,
    rank_videos_by_stat,
)
from personal.basketball_analysis.webapp.backend.tests.test_utils import BackendTestCase

# The repo root's tests/fakes.py holds a stdlib-only duck-typed fake OpenAI-SDK
# client shape that this reuses directly rather than duplicating it. It can't
# be imported as `tests.fakes` here -- webapp/backend/tests is ALSO a package
# named `tests` (per the spec's own directory layout), and whichever one
# Python resolves first via sys.path shadows the other. Loading by explicit
# file path via importlib sidesteps the name collision entirely.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_fakes_path = os.path.join(REPO_ROOT, "tests", "fakes.py")
_spec = importlib.util.spec_from_file_location("_root_tests_fakes", _fakes_path)
_root_fakes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_root_fakes)

FakeClient = _root_fakes.FakeClient
FakeMessage = _root_fakes.FakeMessage
FakeResponse = _root_fakes.FakeResponse
FakeToolCall = _root_fakes.FakeToolCall


def _insert_video(db, video_id, **kwargs):
    defaults = dict(
        filename=f"{video_id}.mp4",
        uploaded_at=datetime.now(timezone.utc),
        upload_path=f"/tmp/{video_id}.mp4",
        status="done",
        has_violations=False,
    )
    defaults.update(kwargs)
    db.add(Video(id=video_id, **defaults))


class TestToolFunctionsDirectly(BackendTestCase):
    def test_find_closest_possession_split(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1", team_a_possession_pct=90.0)
            _insert_video(db, "v2", team_a_possession_pct=51.0)
            db.commit()
            result = find_closest_possession_split(db)
            self.assertEqual(result["video_ids"], ["v2"])
        finally:
            db.close()

    def test_find_closest_possession_split_empty_library(self):
        db = self.db_session()
        try:
            result = find_closest_possession_split(db)
            self.assertEqual(result["video_ids"], [])
            self.assertIsNone(result["result"])
        finally:
            db.close()

    def test_find_videos_by_min_distance(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1")
            _insert_video(db, "v2")
            db.add(Player(video_id="v1", tracker_player_id=1, label="P1", team=1,
                           total_distance_m=500.0, avg_speed_kmh=5.0, max_speed_kmh=10.0))
            db.add(Player(video_id="v2", tracker_player_id=1, label="P1", team=1,
                           total_distance_m=50.0, avg_speed_kmh=5.0, max_speed_kmh=10.0))
            db.commit()
            result = find_videos_by_min_distance(db, min_meters=200.0)
            self.assertEqual(result["video_ids"], ["v1"])
        finally:
            db.close()

    def test_rank_videos_by_stat_valid(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1", total_passes=5)
            _insert_video(db, "v2", total_passes=20)
            db.commit()
            result = rank_videos_by_stat(db, stat="total_passes", order="desc", limit=5)
            self.assertEqual(result["video_ids"], ["v2", "v1"])
        finally:
            db.close()

    def test_rank_videos_by_stat_rejects_unknown_stat(self):
        db = self.db_session()
        try:
            result = rank_videos_by_stat(db, stat="not_a_real_stat")
            self.assertIn("error", result)
            self.assertEqual(result["video_ids"], [])
        finally:
            db.close()

    def test_compare_videos(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1", total_passes=5)
            _insert_video(db, "v2", total_passes=20)
            db.commit()
            result = compare_videos(db, video_id_a="v1", video_id_b="v2")
            self.assertEqual(set(result["video_ids"]), {"v1", "v2"})
            self.assertIn("v1", result["result"])
            self.assertIn("v2", result["result"])
        finally:
            db.close()

    def test_compare_videos_missing_id(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1")
            db.commit()
            result = compare_videos(db, video_id_a="v1", video_id_b="does-not-exist")
            self.assertIn("error", result)
        finally:
            db.close()

    def test_find_videos_with_violations_empty_table_returns_empty_list_not_error(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1")
            db.commit()
            result = find_videos_with_violations(db)
            self.assertNotIn("error", result)
            self.assertEqual(result["result"], [])
            self.assertEqual(result["video_ids"], [])
        finally:
            db.close()


class TestQueryLibraryDispatch(BackendTestCase):
    def test_dispatches_to_rank_videos_by_stat_and_returns_matched_ids(self):
        db = self.db_session()
        try:
            _insert_video(db, "v1", total_passes=5)
            _insert_video(db, "v2", total_passes=20)
            db.commit()

            tool_call = FakeToolCall(
                id="call_1",
                name="rank_videos_by_stat",
                arguments_json=json.dumps({"stat": "total_passes", "order": "desc", "limit": 1}),
            )
            first_response = FakeResponse(FakeMessage(content=None, tool_calls=[tool_call]))
            final_response = FakeResponse(FakeMessage(content="Video v2 had the most passes."))
            client = FakeClient([first_response, final_response])

            result = query_library(client, "fake-model", "which video had the most passes?", db)

            self.assertEqual(result["answer"], "Video v2 had the most passes.")
            self.assertEqual(result["matched_video_ids"], ["v2"])
            self.assertEqual(len(client.calls), 2)
        finally:
            db.close()

    def test_no_tool_call_returns_direct_answer(self):
        db = self.db_session()
        try:
            response = FakeResponse(FakeMessage(content="This is a qualitative answer."))
            client = FakeClient([response])
            result = query_library(client, "fake-model", "tell me about basketball in general", db)
            self.assertEqual(result["answer"], "This is a qualitative answer.")
            self.assertEqual(result["matched_video_ids"], [])
        finally:
            db.close()

    def test_unknown_tool_name_does_not_crash(self):
        db = self.db_session()
        try:
            hallucinated_call = FakeToolCall(
                id="call_1", name="not_a_real_tool", arguments_json=json.dumps({})
            )
            first_response = FakeResponse(FakeMessage(content=None, tool_calls=[hallucinated_call]))
            final_response = FakeResponse(FakeMessage(content="Sorry, I couldn't find that."))
            client = FakeClient([first_response, final_response])

            result = query_library(client, "fake-model", "some question", db)
            self.assertEqual(result["answer"], "Sorry, I couldn't find that.")
        finally:
            db.close()
