.PHONY: install dev test lint format db-up db-down migrate clean

# Install all dependencies (including dev)
install:
	uv sync --all-groups

# Run the FastAPI dev server with hot-reload
dev:
	uv run uvicorn ai_intel.api.main:app --reload --host 0.0.0.0 --port 8000

# Run the test suite
test:
	uv run pytest -v

# Lint with ruff
lint:
	uv run ruff check .

# Format with black + ruff import sorting
format:
	uv run black .
	uv run ruff check --select I --fix .

# Start only the Postgres container
db-up:
	docker compose up -d db

# Stop and remove Postgres container (data volume preserved)
db-down:
	docker compose stop db

# Apply all pending Alembic migrations
migrate:
	uv run alembic upgrade head

# Tear down everything and remove Python cache
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
