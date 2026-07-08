"""
run_pipeline_job: the arq task that actually drives the CV pipeline for one
Job row.

Threading model:
  - arq itself runs an asyncio event loop shared by all concurrently-running
    jobs (up to WorkerSettings.max_jobs).
  - run_analysis() is synchronous, CPU/GPU-bound code -- calling it directly
    would block that shared event loop for the job's entire duration (seconds
    to minutes), starving every other job's progress-publishing. It's run via
    asyncio.to_thread() so the event loop stays responsive.
  - progress_callback is invoked by run_analysis() FROM that worker thread
    (not the event loop thread), synchronously, once per stage boundary. It
    publishes to Redis using a plain sync `redis` client (redis.asyncio's
    client is not thread-safe to call from a non-event-loop thread without
    extra coordination) and updates the Job row's stage columns via a fresh
    sync Session per call -- cheap enough at 13 calls total per job, and
    avoids sharing a Session across threads.
  - DB reads/writes for the job's own lifecycle (load, mark processing/done/
    failed) go through asyncio.to_thread() wrapping a plain sync Session too,
    for the same "don't block the shared event loop on I/O" reason, even
    though SQLite I/O is comparatively fast.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.config import settings
from personal.basketball_analysis.webapp.backend.app.db.base import SessionLocal
from personal.basketball_analysis.webapp.backend.app.db.models import Job, Video
from personal.basketball_analysis.webapp.backend.app.services.report_indexer import index_report
from personal.basketball_analysis.webapp.backend.app.services.storage import get_paths

PROGRESS_CHANNEL_TEMPLATE = "job:{job_id}:progress"


def _publish_progress(job_id: str, payload: Dict[str, Any]) -> None:
    """
    Synchronous Redis publish, safe to call from the worker thread that
    run_analysis()'s progress_callback executes on (NOT the arq event loop
    thread). A fresh connection per call is deliberately simple over pooling
    -- 13 calls per job is not a rate that needs pooling.
    """
    import redis as sync_redis

    client = sync_redis.from_url(settings.REDIS_URL)
    try:
        client.publish(PROGRESS_CHANNEL_TEMPLATE.format(job_id=job_id), json.dumps(payload))
    finally:
        client.close()


def _load_job_and_video(db: Session, job_id: str):
    job = db.get(Job, job_id)
    if job is None:
        raise ValueError(f"No Job row with id {job_id!r}")
    video = db.get(Video, job.video_id)
    if video is None:
        raise ValueError(f"No Video row with id {job.video_id!r} for job {job_id!r}")
    return job, video


def _mark_processing_sync(job_id: str) -> None:
    db = SessionLocal()
    try:
        job, video = _load_job_and_video(db, job_id)
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        video.status = "processing"
        db.commit()
    finally:
        db.close()


def _on_progress_sync(job_id: str, stage_name: str, stage_index: int, total_stages: int, status: str) -> None:
    """
    Called synchronously by run_analysis()'s ProgressReporter, once per stage
    start/end. Updates the Job row (poll fallback) and publishes to Redis
    (SSE fan-out) -- both cheap, both safe to do inline on the worker thread.
    """
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is not None:
            job.stage = stage_name
            job.stage_index = stage_index
            job.total_stages = total_stages
            db.commit()
    finally:
        db.close()

    _publish_progress(
        job_id,
        {
            "stage": stage_name,
            "stage_index": stage_index,
            "total_stages": total_stages,
            "status": status,
        },
    )


def _run_pipeline_sync(job_id: str, video_id: str, upload_path: str) -> Dict[str, Any]:
    """
    Everything that must happen on the worker thread: build the config, run
    run_analysis() with a progress callback wired to this job, return the
    report dict. Raises on failure -- the caller (run_pipeline_job) is
    responsible for catching and marking the job/video failed.
    """
    from personal.basketball_analysis.pipeline import PipelineConfig, run_analysis

    paths = get_paths(video_id)

    def progress_callback(stage_name, stage_index, total_stages, status):
        _on_progress_sync(job_id, stage_name, stage_index, total_stages, status)

    config = PipelineConfig(
        video_path=upload_path,
        output_video_path=paths.output_path,
        report_path=paths.report_path,
        use_stubs=False,
    )
    return run_analysis(config, progress_callback=progress_callback)


def _finalize_success_sync(job_id: str, video_id: str, report: Dict[str, Any]) -> None:
    db = SessionLocal()
    try:
        job, video = _load_job_and_video(db, job_id)
        paths = get_paths(video_id)
        job.status = "done"
        job.finished_at = datetime.now(timezone.utc)
        video.status = "done"
        video.output_path = paths.output_path
        video.report_json_path = paths.report_path if os.path.isfile(paths.report_path) else None
        db.commit()
        index_report(db, video_id, report)
    finally:
        db.close()


def _finalize_failure_sync(job_id: str, error_message: str) -> None:
    db = SessionLocal()
    try:
        job, video = _load_job_and_video(db, job_id)
        job.status = "failed"
        job.error_message = error_message
        job.finished_at = datetime.now(timezone.utc)
        video.status = "failed"
        video.error_message = error_message
        db.commit()
    finally:
        db.close()


async def run_pipeline_job(ctx, job_id: str) -> None:
    """The arq task entrypoint. Signature matches arq's (ctx, *args) contract."""
    db = SessionLocal()
    try:
        job, video = _load_job_and_video(db, job_id)
        video_id = video.id
        upload_path = video.upload_path
    finally:
        db.close()

    await asyncio.to_thread(_mark_processing_sync, job_id)

    try:
        report = await asyncio.to_thread(_run_pipeline_sync, job_id, video_id, upload_path)
    except Exception as exc:  # noqa: BLE001 -- any pipeline failure must be recorded, not swallowed
        error_message = str(exc)
        await asyncio.to_thread(_finalize_failure_sync, job_id, error_message)
        _publish_progress(job_id, {"status": "failed", "error_message": error_message, "terminal": True})
        return

    await asyncio.to_thread(_finalize_success_sync, job_id, video_id, report)
    _publish_progress(job_id, {"stage": "report", "status": "done", "terminal": True})
