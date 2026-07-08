"""
ingestion.py: orchestrates the upload -> DB rows -> enqueue flow.

Split out from the videos route handler so it's independently testable and
mockable -- test_video_routes.py patches enqueue_pipeline_job() so upload
tests never need a real Redis/arq worker.
"""

import uuid
from datetime import datetime, timezone
from typing import BinaryIO, Tuple

from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.config import settings
from personal.basketball_analysis.webapp.backend.app.db.models import Job, Video
from personal.basketball_analysis.webapp.backend.app.services.storage import save_upload


async def enqueue_pipeline_job(job_id: str) -> None:
    """
    Enqueue the arq task that actually runs the pipeline for this job.

    A thin, separately-importable wrapper around arq's redis pool + enqueue
    call specifically so tests can patch *this* function (via
    `unittest.mock.patch("app.services.ingestion.enqueue_pipeline_job")`)
    without needing a real Redis connection.
    """
    from arq import create_pool
    from arq.connections import RedisSettings

    redis = await create_pool(RedisSettings.from_dsn(settings.REDIS_URL))
    try:
        await redis.enqueue_job("run_pipeline_job", job_id)
    finally:
        await redis.close()


async def ingest_upload(db: Session, filename: str, file_obj: BinaryIO) -> Tuple[Video, Job]:
    """
    Full ingestion flow for one uploaded file:
      1. Generate a video_id, save the file to disk.
      2. Create Video (status="queued") + Job (status="queued") rows.
      3. Enqueue the arq job that will process it.

    Returns the (Video, Job) ORM rows, already committed.
    """
    video_id = uuid.uuid4().hex
    paths = save_upload(video_id, filename, file_obj)

    now = datetime.now(timezone.utc)
    video = Video(
        id=video_id,
        filename=filename,
        uploaded_at=now,
        upload_path=paths.upload_path,
        status="queued",
        has_violations=False,
    )
    db.add(video)

    job_id = uuid.uuid4().hex
    job = Job(
        id=job_id,
        video_id=video_id,
        status="queued",
        created_at=now,
    )
    db.add(job)

    db.commit()
    db.refresh(video)
    db.refresh(job)

    await enqueue_pipeline_job(job_id)

    return video, job
