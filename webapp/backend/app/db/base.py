"""
SQLAlchemy engine/session setup -- SYNC, not async.

Deliberate simplification for a local, single-user portfolio app: SQLite has
no async driver story worth the added complexity here (aiosqlite exists but
buys nothing at this scale), and FastAPI already runs sync route handlers in
a thread pool so this never blocks the event loop in a way that matters at
real usage volumes. See webapp README for the full rationale.
"""

from sqlalchemy import create_engine, event
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

# Enable WAL journal mode and a 5-second busy timeout for SQLite.
# WAL allows concurrent readers while one writer holds the lock; busy_timeout
# makes writers retry instead of failing instantly with "database is locked".
# Both are no-ops for PostgreSQL (the event only fires for SQLite connections).
if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
