"""
Report route tests, focused on the narrative/Q&A endpoints' error handling.

Full happy-path narrative/Q&A generation needs a real (or scripted-fake) LLM
client, which game_qa.py's own test suite already covers against tests/fakes.py
at the repo root -- these tests instead cover what the API layer does when the
server has no LLM provider configured at all, since that's a real, previously
broken path: get_video_narrative()/ask_video_question() used to let
llm_client.MissingCredentialError bubble up as an unhandled 500 with a raw
Python traceback instead of a clean, actionable error.
"""

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

from personal.basketball_analysis.webapp.backend.app.db.models import Video
from personal.basketball_analysis.webapp.backend.tests.test_utils import BackendTestCase


class TestNarrativeWithoutLlmCredentials(BackendTestCase):
    def _insert_done_video_with_report(self, video_id="v1"):
        report_path = os.path.join(self.tmp_dir, "reports", f"{video_id}.json")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"players": [], "team_possession": {}}, f)

        db = self.db_session()
        try:
            db.add(
                Video(
                    id=video_id,
                    filename="game.mp4",
                    uploaded_at=datetime.now(timezone.utc),
                    upload_path=f"/tmp/{video_id}.mp4",
                    status="done",
                    report_json_path=report_path,
                    has_violations=False,
                )
            )
            db.commit()
        finally:
            db.close()

    def test_narrative_returns_503_not_500_when_no_llm_key_configured(self):
        self._insert_done_video_with_report()

        with patch("personal.basketball_analysis.webapp.backend.app.services.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_API_KEY = None
            mock_settings.LLM_MODEL = None
            mock_settings.LLM_BASE_URL = None

            response = self.client.get("/api/videos/v1/narrative")

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM_API_KEY", response.json()["detail"])

    def test_qa_returns_503_not_500_when_no_llm_key_configured(self):
        self._insert_done_video_with_report()

        with patch("personal.basketball_analysis.webapp.backend.app.services.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_API_KEY = None
            mock_settings.LLM_MODEL = None
            mock_settings.LLM_BASE_URL = None

            response = self.client.post("/api/videos/v1/qa", json={"question": "Who scored?"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM_API_KEY", response.json()["detail"])

    def test_narrative_returns_cached_text_without_needing_llm_client(self):
        # Already-generated narrative must short-circuit before ever touching
        # the LLM client -- confirms the 503 path above only fires on a real
        # cache miss, not unconditionally.
        video_id = "v1"
        self._insert_done_video_with_report(video_id)
        db = self.db_session()
        try:
            video = db.get(Video, video_id)
            video.narrative_text = "Team A dominated the boards."
            db.commit()
        finally:
            db.close()

        response = self.client.get(f"/api/videos/{video_id}/narrative")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["narrative"], "Team A dominated the boards.")
        self.assertTrue(body["cached"])


if __name__ == "__main__":
    import unittest

    unittest.main()
