# Basketball Analysis -- Web Platform

A FastAPI backend that wraps the existing CV pipeline (`pipeline.run_analysis`)
into a video-upload-and-analyze web platform: upload a game clip, watch it get
annotated and analyzed in the background with live progress, browse a library
of past analyses with filters/sort and natural-language search, and ask
questions about any game (or the whole library) via an LLM tool-calling layer
that never does arithmetic itself -- see `backend/app/llm/library_qa.py` and
the repo root's `game_qa.py` for why.

## Prerequisites

- The root repo's `.venv` already set up (torch/ultralytics/etc. installed) --
  this backend imports `pipeline.run_analysis` directly as a real Python
  import, so it must run in that SAME venv, not a separate one.
- Redis installed (`brew install redis`).
- The three model weight files in `models/*.pt` are **not required** to run
  the API/DB/upload/library endpoints -- only to actually process a video
  end-to-end. See the root README for how to obtain them; until then, upload
  and library browsing work fine, but a queued job will fail once it reaches
  the tracker stages.

## Local run

All commands below assume you start from the repo root
(`basketball_analysis/`).

### 1. Activate the shared venv and install backend dependencies

```bash
source .venv/bin/activate
pip install -r webapp/backend/requirements.txt
```

### 2. Configure environment variables (optional but needed for LLM features)

```bash
cp webapp/.env.example webapp/backend/.env
# edit webapp/backend/.env and set LLM_API_KEY (and LLM_PROVIDER/LLM_MODEL if
# not using the OpenAI default) -- narrative/Q&A/search endpoints need this,
# everything else (upload, library, job status) works without it.
```

### 3. Run database migrations

```bash
cd webapp/backend
alembic upgrade head
```

This creates `webapp/backend/data/app.db` (SQLite) with the `videos`, `jobs`,
`players`, `game_events`, and `violations` tables.

### 4. Start Redis

```bash
brew services start redis
# or, for a one-off foreground-free start:
redis-server --daemonize yes
```

### 5. Start the arq worker (in its own terminal, from `webapp/backend/`)

```bash
source ../../.venv/bin/activate   # if not already active in this terminal
arq app.worker.arq_settings.WorkerSettings
```

This is the process that actually runs `run_analysis()` for each queued
video. Without it running, uploads will sit in `status="queued"` forever.

### 6. Start the API server (in another terminal, from `webapp/backend/`)

```bash
source ../../.venv/bin/activate   # if not already active in this terminal
uvicorn app.main:app --reload --port 8000
```

The API is now at `http://localhost:8000` (interactive docs at
`http://localhost:8000/docs`).

## What works without model weights

The pipeline needs `models/*.pt` (player/ball/court detectors) to actually
process a video -- those aren't in this repo (see root README). Until they're
in place:

- `POST /api/videos` (upload), `GET /api/videos` (library list/filter/sort),
  `GET /api/videos/{id}` (detail), `DELETE /api/videos/{id}` all work fully.
- `GET /api/jobs/{id}` and `GET /api/jobs/{id}/stream` work, but a job will
  transition to `status="failed"` once the worker actually reaches the
  tracker stages (no model to load).
- Report/narrative/Q&A/search/events endpoints all correctly return 404/400
  for videos that never reached `status="done"`, since there's no report to
  operate on yet.

Once real model weights are in place, the full pipeline runs end-to-end and
every endpoint becomes fully exercisable.

## Architecture notes

- **Sync SQLAlchemy, not async.** SQLite has no async driver story worth the
  complexity at this local-single-user scale (aiosqlite exists but buys
  nothing here), and FastAPI runs sync route handlers in a thread pool
  automatically -- this never blocks the event loop in a way that matters at
  real usage volumes. See `app/db/base.py`.
- **arq + Redis for the background job**, not an in-process background task --
  `run_analysis()` is a real, synchronous, CPU/GPU-bound function that can
  take minutes; arq gives durable job state, a real worker process boundary,
  and a `max_jobs` cap so a laptop doesn't try to run 10 pipelines at once.
- **SSE (Server-Sent Events) for live progress**, backed by Redis pub/sub --
  the worker publishes one event per pipeline stage boundary (13 stages, see
  `pipeline/progress.py`'s `STAGE_NAMES`), and `GET /api/jobs/{id}/stream`
  relays them. `GET /api/jobs/{id}` (plain polling) reads the same underlying
  state from the `Job` row for clients that don't want an SSE connection.
- **LLM calls are always on-demand**, never inside the worker/job path.
  `GET /api/videos/{id}/narrative` calls the LLM lazily on first request only
  (cached into `Video.narrative_text` afterward); Q&A and search are live
  per-request by nature. This keeps the actual CV pipeline job independent of
  any external LLM API being reachable, and avoids burning API quota
  narrating videos nobody looks at.
- **Every computed number in an LLM answer traces back to a real function
  call**, never the model's own arithmetic -- see `app/llm/library_qa.py`'s
  docstring and the root `game_qa.py` for the full rationale (models are
  measurably worse at in-context arithmetic than delegating to real code).

## Known gap: no per-frame pass/interception timeline

`GET /api/videos/{id}/events` only returns violation events (double dribble /
traveling), which DO carry real `start_frame`/`end_frame` from the pipeline's
`rule_violation_detector` output. The current `game_report.py` only reports
pass/interception counts aggregated per team, not a timestamped per-frame
event list -- there's no real frame-numbered data to build pass/interception
timeline entries from without a pipeline-side change (surfacing frame-tagged
events in `game_report.py`), which is out of scope for this backend. See the
code comment on `GameEvent` in `app/db/models.py` and the events route in
`app/api/routes/reports.py` for the full explanation.

## Running tests

Backend tests (separate from, and independent of, the CV pipeline's own test
suite at the repo root):

```bash
cd basketball_analysis
source .venv/bin/activate
python3 -m unittest discover -s webapp/backend/tests -t webapp/backend
```

Mocks `pipeline.run_analysis` (never actually invokes the real CV pipeline --
there are no model weights to run it with anyway) and, where a live Redis
would otherwise be needed, either mocks the arq enqueue call or skips
gracefully with a clear reason if Redis isn't reachable on `localhost:6379`.

Root pipeline test suite (must stay green, untouched by this backend work):

```bash
python3 -m unittest discover -s tests -t .
```
