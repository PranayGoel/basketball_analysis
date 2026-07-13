# Build context: repo root  (docker build -f webapp/backend/Dockerfile .)
#
# Import namespace convention: all code uses personal.basketball_analysis.*
# imports. config.py adds sys.path[4 dirs up from itself] = /workspace, so the
# import resolves as /workspace/personal/basketball_analysis/... — which is
# exactly where we place the repo (COPY . /workspace/personal/basketball_analysis/).
# No pyproject.toml gymnastics required; the layout itself is the package.
#
# CMD starts uvicorn.  For the arq worker, override CMD in docker-compose:
#   command: arq app.worker.arq_settings.WorkerSettings

FROM python:3.9-slim

# System libs needed by opencv-python-headless and PyTorch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Layer 1: pipeline dependencies (heavy: torch, ultralytics, etc.)
# Copy requirements before source so Docker caches this layer across source changes.
COPY requirements.txt /tmp/pipeline-requirements.txt
RUN pip install --no-cache-dir -r /tmp/pipeline-requirements.txt

# ── Layer 2: webapp backend dependencies
COPY webapp/backend/requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt

# ── Layer 3: place the repo at the path the import namespace expects
COPY . /workspace/personal/basketball_analysis/

WORKDIR /workspace/personal/basketball_analysis/webapp/backend

EXPOSE 8000

# Default: run the API server.
# Worker override in docker-compose: arq app.worker.arq_settings.WorkerSettings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
