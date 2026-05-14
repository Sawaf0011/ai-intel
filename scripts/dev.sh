#!/usr/bin/env bash
# Convenience: start Postgres and run the FastAPI dev server
set -euo pipefail

echo "Starting Postgres..."
docker compose up -d db

echo "Waiting for Postgres to be healthy..."
until docker compose exec db pg_isready -U aiintel -d aiintel > /dev/null 2>&1; do
    sleep 1
done

echo "Running migrations..."
uv run alembic upgrade head

echo "Starting API..."
uv run uvicorn ai_intel.api.main:app --reload --host 0.0.0.0 --port 8000
