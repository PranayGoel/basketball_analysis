"""
Shared test scaffolding: a temp-file SQLite DB + TestClient wired to override
get_db, matching the repo's plain-unittest style (see tests/fakes.py at the
repo root for the analogous zero-framework-dependency approach used for
llm_client.py/game_qa.py's tests).

Each TestCase using BackendTestCase gets its own throwaway SQLite file under
a tempdir (not the real DATA_DIR/app.db) and its own uploads/outputs/reports
directories, so tests never touch or depend on real runtime state and can run
fully in parallel/isolated from each other.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from personal.basketball_analysis.webapp.backend.app.db.base import Base
from personal.basketball_analysis.webapp.backend.app.db.session import get_db
from personal.basketball_analysis.webapp.backend.app.main import app


class BackendTestCase(unittest.TestCase):
    """
    Base class providing: self.client (TestClient with get_db overridden to
    a fresh temp SQLite DB), self.SessionLocal (for inserting fixture rows
    directly, bypassing the API), and self.tmp_dir (cleaned up in tearDown).

    Also patches app.services.storage.settings' uploads_dir/outputs_dir/
    reports_dir to point inside self.tmp_dir -- any route or service call
    that goes through storage.get_paths()/save_upload()/delete_video_files()
    (e.g. POST /api/videos actually writing the uploaded file) lands in the
    throwaway tempdir, never the real DATA_DIR. Without this, an upload test
    using the real settings singleton would write real files into the dev
    data/uploads/ directory on every run -- this bit exactly that way before
    the patch was added here (see git history / the smoke-test writeup that
    caught it).
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="bball_backend_test_")
        db_path = f"{self.tmp_dir}/test.db"
        self.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        def override_get_db():
            db = self.SessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        self._storage_settings_patch = patch("personal.basketball_analysis.webapp.backend.app.services.storage.settings")
        mock_settings = self._storage_settings_patch.start()
        mock_settings.uploads_dir = os.path.join(self.tmp_dir, "uploads")
        mock_settings.outputs_dir = os.path.join(self.tmp_dir, "outputs")
        mock_settings.reports_dir = os.path.join(self.tmp_dir, "reports")

    def tearDown(self) -> None:
        self._storage_settings_patch.stop()
        app.dependency_overrides.clear()
        self.engine.dispose()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def db_session(self):
        return self.SessionLocal()
