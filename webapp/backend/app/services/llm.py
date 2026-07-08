"""
Shared LLM-client acquisition for every route that calls out to an LLM
(narrative generation, single-game Q&A, library-wide NL search).

Centralized here (rather than duplicated per route module) so the
missing/invalid-provider-config error path -- a foreseeable, expected
deployment state, not a bug -- is handled identically everywhere: a clean
503 with an actionable message, never an unhandled 500 with a raw Python
traceback leaking to the client.
"""

from fastapi import HTTPException

from personal.basketball_analysis.webapp.backend.app.config import settings


def get_llm_client_or_503():
    from personal.basketball_analysis.llm_client import get_client

    try:
        return get_client(
            provider=settings.LLM_PROVIDER,
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "This feature needs an LLM provider configured on the server "
                f"(set LLM_API_KEY / LLM_PROVIDER -- see webapp/.env.example). {exc}"
            ),
        ) from exc
