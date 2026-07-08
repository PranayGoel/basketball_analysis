"""Pydantic response model for job status polling."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class JobStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    video_id: str
    status: str
    stage: Optional[str] = None
    stage_index: Optional[int] = None
    total_stages: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
