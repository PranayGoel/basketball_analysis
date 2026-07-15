"""Pydantic models for library-wide NL search."""

from typing import List

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=2000)


class SearchResponse(BaseModel):
    answer: str
    matched_video_ids: List[str]
