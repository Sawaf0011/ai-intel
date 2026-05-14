# ============================================================
# Stage 1: builder — install Python dependencies with uv
# ============================================================
FROM python:3.12-slim AS builder

# Pull the uv binary from the official Astral image (pins to dev version)
COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /bin/uv

# uv environment flags for Docker builds
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# --- Layer 1: dependency manifests only ---
# Copied before source so this layer is cached independently.
# Rebuilt only when pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock README.md ./

# Install runtime deps without the project itself (cache-friendly)
RUN uv sync --frozen --no-install-project --no-dev

# --- Layer 2: source code ---
COPY src/ ./src/

# Install the project itself into the venv
RUN uv sync --frozen --no-dev


# ============================================================
# Stage 2: runtime — lean image that runs the application
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN groupadd --system app \
    && useradd --system --gid app --create-home --shell /bin/bash app

# Bring in the fully-installed venv from the builder
COPY --from=builder /opt/venv /opt/venv

# Application source (needed for editable-install .pth resolution)
COPY --from=builder /app/src ./src/

# Alembic migration files and config (used by the migrate service)
COPY alembic/ ./alembic/
COPY alembic.ini pyproject.toml uv.lock ./

# Put the venv on PATH
ENV PATH="/opt/venv/bin:$PATH"

# Hand ownership to the non-root user
RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()"

CMD ["uvicorn", "ai_intel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
