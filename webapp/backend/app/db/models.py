"""
ORM models: Video, Job, Player, GameEvent, Violation.

Python 3.9 compatible throughout -- Optional[X]/List[X] from typing, never the
PEP 604 `X | None` union syntax (not available until 3.10).

Video carries a set of flattened analytics columns (team_a/b_possession_pct,
player_count, total_passes, ...) populated by services/report_indexer.py once
a job completes. These exist purely so the library list/filter/sort endpoint
and the NL-search tool functions (llm/library_qa.py) can express queries in
SQL (ORDER BY, WHERE, func.abs(...)) instead of loading every report JSON off
disk and filtering in Python for every request.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from personal.basketball_analysis.webapp.backend.app.db.base import Base


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # uuid4 hex
    filename: Mapped[str] = mapped_column(String)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upload_path: Mapped[str] = mapped_column(String)
    output_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String)  # queued|processing|done|failed
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Flattened analytics -- populated by report_indexer after job completion.
    team_a_possession_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    team_b_possession_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    player_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_passes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_interceptions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    max_player_speed_kmh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    max_player_distance_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    has_violations: Mapped[bool] = mapped_column(Boolean, default=False)
    narrative_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    report_json_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    jobs: Mapped[list] = relationship("Job", back_populates="video", cascade="all, delete-orphan")
    players: Mapped[list] = relationship("Player", back_populates="video", cascade="all, delete-orphan")
    game_events: Mapped[list] = relationship("GameEvent", back_populates="video", cascade="all, delete-orphan")
    violations: Mapped[list] = relationship("Violation", back_populates="video", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.id"))
    status: Mapped[str] = mapped_column(String)  # queued|processing|done|failed
    stage: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    stage_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_stages: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="jobs")


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.id"), index=True)
    tracker_player_id: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String)
    team: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_distance_m: Mapped[float] = mapped_column(Float)
    avg_speed_kmh: Mapped[float] = mapped_column(Float)
    max_speed_kmh: Mapped[float] = mapped_column(Float)

    video: Mapped["Video"] = relationship("Video", back_populates="players")


class GameEvent(Base):
    """
    Aggregate per-team pass/interception counts, one row per (team, type) pair
    the report reports on -- NOT a timestamped per-event list.

    Honesty note: game_report.py's `events` key only carries aggregate counts
    (events.passes.team_1, etc.), not a per-frame event list with frame
    numbers. There is currently no pipeline-side data to derive real
    timestamps from for passes/interceptions. GameEvent rows here exist so
    that count-based library queries (rank_videos_by_stat("total_passes"))
    can be expressed as a simple SQL aggregate without re-parsing report JSON,
    but /api/videos/{id}/events (the "smart timeline") intentionally excludes
    these -- see api/routes/reports.py's events handler for that decision.
    Surfacing real per-frame pass/interception events would require a
    pipeline-side change (frame-tagged events in game_report.py), which is
    out of scope for this backend task.
    """

    __tablename__ = "game_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.id"), index=True)
    event_type: Mapped[str] = mapped_column(String)  # "pass" | "interception"
    team: Mapped[int] = mapped_column(Integer)

    video: Mapped["Video"] = relationship("Video", back_populates="game_events")


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(String, ForeignKey("videos.id"), index=True)
    player_id: Mapped[int] = mapped_column(Integer)
    start_frame: Mapped[int] = mapped_column(Integer)
    end_frame: Mapped[int] = mapped_column(Integer)
    violation_type: Mapped[str] = mapped_column(String)
    confidence: Mapped[str] = mapped_column(String)

    video: Mapped["Video"] = relationship("Video", back_populates="violations")
