"""
Report routes: full report fetch, lazy/cached narrative generation,
single-game tool-calling Q&A, and the violations-only event timeline.

LLM calls here are always on-demand, triggered directly by a user action
(viewing a video's narrative for the first time, or asking a question) --
never from the job/worker path. See worker/tasks.py's docstring and the
webapp README for the "why" (no external-API dependency in the pipeline job,
no wasted quota narrating videos nobody looks at).
"""

import json
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.api.deps import get_db
from personal.basketball_analysis.webapp.backend.app.db.models import Video, Violation
from personal.basketball_analysis.webapp.backend.app.schemas.report import EventsResponse, NarrativeResponse, QARequest, QAResponse, ViolationEvent
from personal.basketball_analysis.webapp.backend.app.services.llm import get_llm_client_or_503

router = APIRouter(prefix="/api", tags=["reports"])


def _load_report_or_404(video: Video) -> dict:
    if not video.report_json_path or not os.path.isfile(video.report_json_path):
        raise HTTPException(status_code=404, detail="Report not yet available for this video")
    with open(video.report_json_path) as f:
        return json.load(f)


def _get_video_or_404(db: Session, video_id: str) -> Video:
    video = db.get(Video, video_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    return video


@router.get("/videos/{video_id}/report")
def get_video_report(video_id: str, db: Session = Depends(get_db)):
    video = _get_video_or_404(db, video_id)
    return _load_report_or_404(video)


@router.get("/videos/{video_id}/narrative", response_model=NarrativeResponse)
def get_video_narrative(video_id: str, db: Session = Depends(get_db)):
    video = _get_video_or_404(db, video_id)

    if video.narrative_text:
        return NarrativeResponse(narrative=video.narrative_text, cached=True)

    if video.status != "done":
        raise HTTPException(status_code=400, detail="Video has not finished processing yet")

    report = _load_report_or_404(video)

    from personal.basketball_analysis.game_qa import generate_game_narrative

    client, config = get_llm_client_or_503()
    narrative = generate_game_narrative(client, config["model"], report)

    video.narrative_text = narrative
    db.commit()

    return NarrativeResponse(narrative=narrative, cached=False)


@router.post("/videos/{video_id}/qa", response_model=QAResponse)
def ask_video_question(video_id: str, body: QARequest, db: Session = Depends(get_db)):
    video = _get_video_or_404(db, video_id)
    if video.status != "done":
        raise HTTPException(status_code=400, detail="Video has not finished processing yet")

    report = _load_report_or_404(video)

    from personal.basketball_analysis.game_qa import answer_question

    client, config = get_llm_client_or_503()
    answer = answer_question(client, config["model"], report, body.question)
    return QAResponse(answer=answer)


@router.get("/videos/{video_id}/events", response_model=EventsResponse)
def get_video_events(video_id: str, db: Session = Depends(get_db)):
    """
    Violation events only -- the report currently gives pass/interception
    counts aggregated per team, not a timestamped per-event list, so there is
    no real frame-numbered data to build pass/interception timeline entries
    from. Surfacing those would require a pipeline-side change (frame-tagged
    events in game_report.py) that's out of scope for this backend; this
    endpoint intentionally does not fabricate frame numbers to paper over
    that gap. Violation rows DO carry real start_frame/end_frame from the
    pipeline's rule_violation_detector output, so those are safe to return.
    """
    _get_video_or_404(db, video_id)
    violations = db.query(Violation).filter(Violation.video_id == video_id).all()
    events = [
        ViolationEvent(
            player_id=v.player_id,
            start_frame=v.start_frame,
            end_frame=v.end_frame,
            violation_type=v.violation_type,
            confidence=v.confidence,
        )
        for v in violations
    ]
    return EventsResponse(events=events)
