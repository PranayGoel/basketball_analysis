"""
WorkerSettings: the arq entrypoint. Run with:
    cd webapp/backend && arq app.worker.arq_settings.WorkerSettings

max_jobs=2: this is CPU/GPU-bound inference work (real torch/ultralytics
calls inside run_analysis), so there's no benefit to arq's usual
high-concurrency defaults on a single laptop -- 2 keeps one job's own
to_thread() responsive to the event loop while still allowing a second job to
be picked up without waiting for the first to fully finish enqueuing its
final DB writes.
"""

from arq.connections import RedisSettings

from personal.basketball_analysis.webapp.backend.app.config import settings
from personal.basketball_analysis.webapp.backend.app.worker.tasks import run_pipeline_job


class WorkerSettings:
    functions = [run_pipeline_job]
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    max_jobs = 2
    job_timeout = 1800  # 30 min ceiling
