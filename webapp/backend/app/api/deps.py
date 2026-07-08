"""
Shared route dependencies beyond get_db (which lives in app.db.session since
it's a DB-layer concern, not an API-layer one -- re-exported here so routes
have one obvious import path for all their dependencies).
"""

from personal.basketball_analysis.webapp.backend.app.db.session import get_db

__all__ = ["get_db"]
