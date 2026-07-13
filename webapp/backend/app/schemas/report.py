"""Pydantic models for narrative/Q&A/events responses."""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, RootModel


class NarrativeResponse(BaseModel):
    narrative: str
    cached: bool


class QARequest(BaseModel):
    question: str = Field(..., max_length=2000)


class QAResponse(BaseModel):
    answer: str


class ViolationEvent(BaseModel):
    player_id: int
    start_frame: int
    end_frame: int
    violation_type: str
    confidence: str


class EventsResponse(BaseModel):
    events: List[ViolationEvent]


class ReportResponse(RootModel[Dict[str, Any]]):
    """
    Pass-through wrapper: the report JSON on disk is already the exact shape
    game_report.build_game_report() produces, so this just validates it's a
    JSON object without re-declaring every field (players/team_possession/
    events/num_frames/violations?) -- that shape lives in game_report.py and
    this backend must not fork a second definition of it that can drift.

    Pydantic v2's RootModel replaces v1's `__root__` field for this pattern.
    """
