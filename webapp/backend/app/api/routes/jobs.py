"""
Job routes: status poll (fallback for clients without SSE) and the live SSE
progress stream, both reading the same underlying state -- the poll endpoint
reads the Job row's stage columns (updated synchronously by the worker on
every progress callback), the SSE endpoint subscribes to the Redis pub/sub
channel the worker publishes the identical events to.
"""

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from personal.basketball_analysis.webapp.backend.app.api.deps import get_db
from personal.basketball_analysis.webapp.backend.app.config import settings
from personal.basketball_analysis.webapp.backend.app.db.models import Job
from personal.basketball_analysis.webapp.backend.app.schemas.job import JobStatus

router = APIRouter(prefix="/api", tags=["jobs"])

PROGRESS_CHANNEL_TEMPLATE = "job:{job_id}:progress"

# SSE stream gives up waiting for a terminal event after this long, in case a
# worker crashed without publishing one -- the client falls back to polling
# GET /api/jobs/{id} rather than holding an HTTP connection open forever.
_SSE_TIMEOUT_SECONDS = 1800.0


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus.model_validate(job)


@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str, db: Session = Depends(get_db)):
    """
    SSE stream of live progress events, subscribed to the Redis pub/sub
    channel the worker publishes to. Breaks on a terminal event: either
    {"status": "done", "terminal": true} (success) or {"status": "error"}
    (failure) -- both published exactly once by worker/tasks.py at the end
    of run_pipeline_job.
    """
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # A job that's already terminal by the time the client connects (e.g. a
    # page refresh after completion) should get one immediate event and close,
    # rather than subscribing and waiting for an event that will never come.
    if job.status in ("done", "failed"):
        async def already_terminal():
            payload = {
                "status": job.status,
                "terminal": True,
                "stage": job.stage,
                "error_message": job.error_message,
            }
            yield f"event: done\ndata: {json.dumps(payload)}\n\n"

        return StreamingResponse(already_terminal(), media_type="text/event-stream")

    async def event_generator():
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()
        channel = PROGRESS_CHANNEL_TEMPLATE.format(job_id=job_id)
        try:
            await pubsub.subscribe(channel)
            loop = asyncio.get_event_loop()
            deadline = loop.time() + _SSE_TIMEOUT_SECONDS
            while loop.time() < deadline:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    continue
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")

                try:
                    parsed = json.loads(data)
                except (TypeError, ValueError):
                    parsed = {}

                # The browser EventSource API only routes a frame to a named
                # listener (addEventListener("stage"/"done", ...)) when the
                # frame sets a matching `event:` line -- an unnamed `data:`
                # frame defaults to type "message", which the frontend never
                # listens for. Every terminal payload (success or failure)
                # sets "terminal": true; that's what distinguishes "done" from
                # an ordinary per-stage "stage" update here.
                is_terminal = bool(parsed.get("terminal"))
                event_name = "done" if is_terminal else "stage"
                yield f"event: {event_name}\ndata: {data}\n\n"

                if is_terminal:
                    break
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()
            await redis_client.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
