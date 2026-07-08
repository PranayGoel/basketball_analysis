"""
Library-wide NL search route tests, focused on error handling.

Mirrors test_reports_routes.py's narrative/Q&A coverage: search_library()
calls the same shared app.services.llm.get_llm_client_or_503() helper, so it
should degrade the same way (clean 503, not an unhandled 500) when no LLM
provider is configured.
"""

from unittest.mock import patch

from personal.basketball_analysis.webapp.backend.tests.test_utils import BackendTestCase


class TestSearchWithoutLlmCredentials(BackendTestCase):
    def test_search_returns_503_not_500_when_no_llm_key_configured(self):
        with patch("app.services.llm.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_API_KEY = None
            mock_settings.LLM_MODEL = None
            mock_settings.LLM_BASE_URL = None

            response = self.client.post("/api/search", json={"query": "games with the most players"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("LLM_API_KEY", response.json()["detail"])


if __name__ == "__main__":
    import unittest

    unittest.main()
