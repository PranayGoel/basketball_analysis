"""
SQLAlchemy engine/session setup -- SYNC, not async.

Deliberate simplification for a local, single-user portfolio app: SQLite has
no async driver story worth the added complexity here (aiosqlite exists but
buys nothing at this scale), and FastAPI already runs sync route handlers in
a thread pool so this never blocks the event loop in a way that matters at
real usage volumes. See webapp README for the full rationale.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from personal.basketball_analysis.webapp.backend.app.config import settings


class Base(DeclarativeBase):
    pass


# check_same_thread=False is required for SQLite when a Session created on
# one thread is used from another -- FastAPI's threadpool-per-request model
# means that's exactly what happens here, and Session objects created fresh
# per-request (never shared across requests/threads concurrently) is what
# actually keeps this safe, not the flag itself.
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
