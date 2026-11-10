
 
ARG PYTHON_VERSION=3.12
# Pin this to the version you use locally (`uv --version`) so builds are repeatable.
ARG UV_VERSION=latest
 
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
 
# ─────────────────────────────────────────────────────────────────────────────
# base: the same Python for every stage
# ─────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
 
# ─────────────────────────────────────────────────────────────────────────────
# builder: where dependencies are installed. Compilers live HERE and never reach
# the final image.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /uvx /bin/
# Only these two files are copied before installing, so this layer is rebuilt only when the
# dependencies change, not on every code change.
COPY pyproject.toml uv.lock ./
 
# ── backend dependencies (default groups minus dev and frontend) ──
FROM builder AS venv-backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-group frontend --no-install-project
# Clean up pip cache to save room
RUN rm -rf /root/.cache/pip
# Download the embedding model now. The running container then never needs the internet for it.
ENV HF_HOME=/opt/hf
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
# Remove unnecessary files from venv to reduce size.
RUN find /app/.venv -name "*.pyc" -delete \
    && find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /app/.venv -name "*.so.dbg" -delete \
    && find /app/.venv -name "*.dbg" -delete \
    && find /app/.venv -name "LICENSE*" -delete
 
# ── frontend dependencies (Streamlit and requests only) ──
FROM builder AS venv-frontend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --only-group frontend
 
# ─────────────────────────────────────────────────────────────────────────────
# runtime: small, non-root, no compilers, no uv
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS runtime
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app --shell /usr/sbin/nologin app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    HOME=/home/app
 
# ── backend: API, Celery worker, Flower ──
FROM runtime AS backend
# libgomp1: used by PyTorch's CPU math libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
# The code and the virtualenv stay root-owned and read-only for the app user. If the app is ever
# compromised, it cannot rewrite itself.
COPY --from=venv-backend /app/.venv /app/.venv
COPY --from=venv-backend /opt/hf /opt/hf
COPY src ./src
COPY backend ./backend
RUN python -m compileall -q src backend
# Only these folders are writable. docker-compose mounts volumes over them.
RUN mkdir -p /app/uploads /app/data /app/data/chroma_db \
    && chown -R app:app /app/uploads /app/data
ENV HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    UPLOAD_DIR=/app/uploads \
    CHROMA_DIR=/app/data/chroma_db
USER app
EXPOSE 8000
# Used for the API. Celery and Flower turn this off in docker-compose.
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"]
# Exec form, so Docker's stop signal reaches uvicorn and it shuts down cleanly.
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
 
# ── frontend: Streamlit ──
FROM runtime AS frontend
COPY --from=venv-frontend /app/.venv /app/.venv
COPY main.py ./main.py
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    API_BASE_URL=http://backend:8000
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=10s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"]
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]