"""Pydantic response models for the videos routes."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class VideoSummary(BaseModel):
    """Row shape returned by the library list endpoint."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    uploaded_at: datetime
    duration_sec: Optional[float] = None
    status: str
    error_message: Optional[str] = None
    team_a_possession_pct: Optional[float] = None
    team_b_possession_pct: Optional[float] = None
    player_count: Optional[int] = None
    total_passes: Optional[int] = None
    total_interceptions: Optional[int] = None
    max_player_speed_kmh: Optional[float] = None
    max_player_distance_m: Optional[float] = None
    has_violations: bool = False


class VideoDetail(VideoSummary):
    """Full row shape for GET /api/videos/{id} -- adds file paths."""

    upload_path: str
    output_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    narrative_text: Optional[str] = None
    report_json_path: Optional[str] = None


class VideoListResponse(BaseModel):
    items: List[VideoSummary]
    total: int
    page: int
    page_size: int


class UploadResponse(BaseModel):
    video_id: str
    job_id: str
    status: str
