# ─────────────────────────────────────────────────────────────────────────────
# BookRAG — Production Dockerfile
# Multi-stage build: builder installs deps, runtime is a lean deployable image.
#
# Usage:
#   Build:
#     docker build -t bookrag:latest .
#
#   Run (with vectorstore mounted from host):
#     docker run --rm -p 8000:8000 \
#       -v $(pwd)/vectorstore:/app/vectorstore:ro \
#       bookrag:latest
#
#   Health check:
#     curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────


# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
# Uses the official uv image (uv pre-installed) on Debian Bookworm slim.
# Installs only production dependencies into an isolated virtual environment.
# Nothing from this stage leaks into the final image except the .venv directory.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Build-time environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Tell uv where to put the venv
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    # Do not write uv's own cache during Docker build (saves image space)
    UV_NO_CACHE=1 \
    # Install into the project-scoped venv, not the system Python
    UV_SYSTEM_PYTHON=0

WORKDIR /app

# ── Copy dependency manifests only (layer-cache friendly) ──────────────────
# Copying pyproject.toml + uv.lock before source means `uv sync` is only
# re-run when dependencies actually change, not on every source edit.
COPY pyproject.toml uv.lock ./

# ── Install production dependencies (no dev extras) ─────────────────────────
# --frozen  : fail if uv.lock is out of sync with pyproject.toml (safe for CI)
# --no-dev  : exclude dev/test dependencies from the runtime image
RUN uv sync --frozen --no-dev --no-install-project


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
# Starts fresh from the smallest available Python base.
# Only the compiled .venv and application source are copied in — no uv,
# no build tools, no dangling cache directories.
FROM python:3.12-slim-bookworm AS runtime

# ── Runtime environment variables ────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make the venv's bin/ take precedence over system Python
    PATH="/app/.venv/bin:$PATH" \
    # Allow `import src.api.main` and `import config.config` as namespace packages
    PYTHONPATH="/app" \
    # Harden Python against common supply-chain attack vectors
    PYTHONHASHSEED=random \
    # Point HuggingFace cache to a directory inside /app that appuser owns.
    # This prevents the PermissionError on /home/appuser that occurs when
    # useradd --no-create-home is used (default home dir never gets created).
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface"

# ── Create a dedicated non-root user ─────────────────────────────────────────
# Running as root inside a container is a security anti-pattern.
# All files are owned by appuser; the server process inherits the same uid.
RUN groupadd --system --gid 1001 appgroup && \
    useradd  --system --uid 1001 --gid appgroup \
             --no-create-home --shell /sbin/nologin \
             appuser

WORKDIR /app

# ── Copy virtual environment from builder ────────────────────────────────────
COPY --from=builder --chown=appuser:appgroup /app/.venv /app/.venv

# ── Copy application source ───────────────────────────────────────────────────
# Only the directories the runtime actually needs.
# data/ and vectorstore/ are intentionally excluded (see .dockerignore):
#   - data/          : raw source documents; not needed at query-time
#   - vectorstore/   : FAISS index; mounted as a volume at runtime
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup src/    ./src/

# ── Pre-download the HuggingFace embedding model ─────────────────────────────
# Downloads the model weights into /app/.cache/huggingface at BUILD time so:
#   1. Container startup is fast (no network call needed at runtime).
#   2. The image works in air-gapped / restricted-network environments.
#   3. The exact model version is frozen into the image layer.
# Running as root here so the cache directory can be created and chowned.
RUN mkdir -p /app/.cache/huggingface && \
    /app/.venv/bin/python -c \
        "from sentence_transformers import SentenceTransformer; \
         SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', \
         cache_folder='/app/.cache/huggingface')" && \
    chown -R appuser:appgroup /app/.cache

# ── Create the vectorstore mount point ───────────────────────────────────────
# The FAISS index lives here.  Mount the pre-built index at run time with:
#   -v "$(pwd)/vectorstore:/app/vectorstore:ro"
RUN mkdir -p vectorstore/faiss_index && \
    chown -R appuser:appgroup vectorstore/

# Declare the mount point so Docker knows it is intended to hold external data
VOLUME ["/app/vectorstore"]

# ── Drop to non-root ─────────────────────────────────────────────────────────
USER appuser

# ── Expose the application port ──────────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────────────────────────
# Uses Python's built-in urllib — no curl/wget needed in the slim image.
# --start-period=90s  : allow time for FAISS index + embedding model to load.
# --interval=30s      : poll every 30 seconds once the server is live.
# --timeout=10s       : declare unhealthy if the request takes longer.
# --retries=3         : mark unhealthy only after 3 consecutive failures.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
try: \
    urllib.request.urlopen('http://localhost:8000/health', timeout=8); \
    sys.exit(0) \
except Exception: \
    sys.exit(1)"

# ── Start the server ─────────────────────────────────────────────────────────
# Uses exec-form (no shell wrapper) so SIGTERM is delivered directly to
# uvicorn, enabling graceful shutdown of in-flight requests.
#
# --workers 1
#   The RAGPipeline holds the FAISS index and embedding model in memory.
#   Multiple workers would each load their own copy, multiplying RAM usage.
#   Use a reverse proxy (nginx / load balancer) for horizontal scaling instead.
#
# --no-access-log
#   Access logs are handled by the middleware in src/api/main.py with request
#   IDs, so uvicorn's duplicate access log is suppressed.
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]
