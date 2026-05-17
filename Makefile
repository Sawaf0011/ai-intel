.PHONY: install dev test lint format db-up db-down migrate migrate-down revision \
        up up-build down logs logs-app ps migrate-docker clean scrape-github embed

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

# Stop all compose services (volume preserved)
db-down:
	docker compose down

# Apply all pending Alembic migrations
migrate:
	uv run alembic upgrade head

# Roll back one migration
migrate-down:
	uv run alembic downgrade -1

# Generate a new autogenerate migration: make revision m="your message"
revision:
	uv run alembic revision --autogenerate -m "$(m)"

# --- Docker Compose targets ---

# Start the full stack in the background
up:
	docker compose up -d

# Rebuild images and start the full stack
up-build:
	docker compose up -d --build

# Stop and remove all containers (volume preserved)
down:
	docker compose down

# Tail logs from all services
logs:
	docker compose logs -f

# Tail logs from the app service only
logs-app:
	docker compose logs -f app

# Show running compose services and their health
ps:
	docker compose ps

# Run the migrate service manually against the compose db
migrate-docker:
	docker compose run --rm migrate

# Scrape GitHub AI repos into the local database (requires GITHUB_TOKEN in .env)
scrape-github:
	uv run python -m ai_intel.cli scrape --source github $(if $(since),--since $(since),)

# Generate vector embeddings for all un-embedded items
embed:
	uv run python -m ai_intel.cli embed

# Tear down everything and remove Python cache
clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
