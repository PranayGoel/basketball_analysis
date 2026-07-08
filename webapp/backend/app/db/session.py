"""FastAPI dependency yielding a Session, closed after the request."""

from typing import Generator

from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.db.base import SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
