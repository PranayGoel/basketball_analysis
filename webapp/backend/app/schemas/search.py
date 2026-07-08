"""Pydantic models for library-wide NL search."""

from typing import List

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    answer: str
    matched_video_ids: List[str]
