"""
Settings: every runtime knob the backend needs, resolved from environment
variables (or a `.env` file in webapp/backend/, see .env.example at the
webapp/ root) via pydantic-settings.

DATA_DIR is the one directory the backend owns end to end -- uploads, pipeline
outputs, and the SQLite file all live under it, so wiping it is a full reset
with no leftover state elsewhere. Kept out of git (see root .gitignore).

This module also puts the CV pipeline repo root on sys.path -- it's imported
by literally everything else in this app (main.py, worker/tasks.py,
alembic/env.py, every route module transitively via app.db.base), so doing it
here, at import time, guarantees `import pipeline`, `from game_qa import ...`,
and `from llm_client import ...` all resolve correctly regardless of which
directory the process was launched from (`uvicorn`/`arq`/`alembic` are all
documented to run from webapp/backend/, which does NOT put the repo root
-- two directories up -- on sys.path on its own).
"""

import os
import sys
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config.py -> app/ -> backend/ -> webapp/ -> repo root (4 levels up).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Root directory for all runtime state: uploads/, outputs/, reports/,
    # thumbnails/, and app.db. Relative paths are resolved against this
    # module's directory (webapp/backend/app/../data), not the process cwd,
    # so `uvicorn app.main:app` and `arq app.worker.arq_settings.WorkerSettings`
    # agree on the same location regardless of which directory they're launched
    # from.
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    DATABASE_URL: Optional[str] = None  # defaults to sqlite under DATA_DIR if unset
    REDIS_URL: str = "redis://localhost:6379"

    # CORS: the Vite dev server default port. A future frontend build can add
    # its own origin here without touching route code.
    CORS_ORIGINS: str = "http://localhost:5173"

    # Passed through to llm_client.get_client() by the LLM routes/library_qa --
    # this app never hardcodes a provider or key, it just forwards these.
    LLM_PROVIDER: str = "openai"
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    LLM_BASE_URL: Optional[str] = None

    # Maximum size in MB accepted for video uploads. Requests exceeding this
    # limit are rejected with 413 before any bytes are written to disk.
    MAX_UPLOAD_SIZE_MB: int = 500

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        db_path = os.path.join(self.DATA_DIR, "app.db")
        return f"sqlite:///{db_path}"

    @property
    def uploads_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "uploads")

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "outputs")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "reports")

    @property
    def thumbnails_dir(self) -> str:
        return os.path.join(self.DATA_DIR, "thumbnails")

    @property
    def cors_origins_list(self):
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


settings = Settings()
