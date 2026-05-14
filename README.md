# AI Startup Intelligence & Observability Platform

A data pipeline and RAG system that monitors the AI startup ecosystem across five
sources — GitHub, Reddit, Product Hunt, Y Combinator, and Hacker News — and exposes
the collected intelligence through a semantic search chatbot, an agent with tool
calling, and an MCP server. A built-in observability layer surfaces retrieval scores,
token costs, latency, and the reasoning path behind every query.

## What This Is

This project is a portfolio backend that answers one question: *what's happening in
the AI startup space right now?* It ingests signals from five curated sources daily,
stores them in Postgres with pgvector, and lets you query them conversationally via
RAG. The agent layer adds structured tool calls — search by keyword, surface trending
items, compare sources, summarise a thread — so the chatbot can reason across the
data rather than just retrieve it.

The same tools are exposed as an MCP server, meaning any MCP-compatible client
(Claude Desktop, etc.) can call them directly. The observability dashboard shows
exactly what the retrieval pipeline did for each query: which chunks were scored,
how much the LLM call cost, and the full reasoning chain.

**Current state:** the backend foundation is complete — database, API skeleton,
Docker stack, migrations. Feature work (scrapers, embeddings, RAG, agent) starts
next. See the [Roadmap](#roadmap).

## Current Status

| Component | Status |
|---|---|
| Project skeleton, tooling (uv, ruff, pytest) | ✅ Done |
| FastAPI app with `/health` endpoint | ✅ Done |
| Postgres 16 + pgvector schema, Alembic migrations | ✅ Done |
| Docker + docker-compose (db → migrate → app) | ✅ Done |
| CI workflow scaffold | ✅ Done |
| Source scrapers (GitHub, Reddit, Product Hunt, YC, HN) | 🚧 GitHub done; others planned |
| Embedding pipeline and vector search | 🚧 Planned |
| RAG-powered chat with citations | 🚧 Planned |
| Agent with tool calling (search, trending, compare, summarise) | 🚧 Planned |
| MCP server exposing the agent tools | 🚧 Planned |
| Observability dashboard (scores, cost, latency, reasoning paths) | 🚧 Planned |
| Automated daily digest | 🚧 Planned |

## Tech Stack

- **Python 3.12** — primary language
- **FastAPI** — async HTTP framework; lifespan-managed startup/shutdown
- **SQLAlchemy 2.x (async)** — ORM with full async session support via asyncpg
- **Postgres 16 + pgvector** — primary store; vector(1536) column for embeddings
- **Alembic** — database migrations with async engine support
- **uv** — dependency management and lockfile; replaces pip/pip-tools/venv
- **Docker + docker-compose** — containerised stack with healthchecks and a
  dedicated one-shot migrate service
- **OpenAI SDK** — chat completions (gpt-4o-mini) and embeddings
  (text-embedding-3-small)
- **mcp** — Anthropic's official Python MCP SDK (FastMCP)
- **Pydantic v2 + pydantic-settings** — request/response models and typed config
- **ruff + black** — linting and formatting
- **pytest + pytest-asyncio** — async test suite

## Architecture

The docker-compose stack runs three services in strict dependency order:

```
pgvector/pgvector:pg16          python:3.12-slim (built from Dockerfile)
        │                              │                    │
      [ db ]  ── healthy ──►  [ migrate ]  ── exit(0) ──►  [ app ]
   port 5432                  alembic upgrade head       port 8001→8000
   named volume
```

- **db** — Postgres 16 with the pgvector extension pre-installed. Exposes port 5432
  for direct host access (`psql`, `alembic` run locally).
- **migrate** — same image as `app`; runs `alembic upgrade head` then exits.
  Idempotent: re-running the stack after migrations are applied is a no-op.
- **app** — FastAPI served by uvicorn. Starts only after `migrate` exits with code 0.

The app image is a two-stage build: a `builder` stage installs dependencies into
`/opt/venv` via uv, and the `runtime` stage copies only the venv and source —
no build tools in the final image.

## Quickstart (Docker — recommended)

> Prerequisites: Docker Desktop

```bash
git clone <repo-url> && cd ai-intel
cp .env.example .env        # then open .env and set OPENAI_API_KEY
docker compose up -d --build
curl http://localhost:8001/health
# → {"status":"ok","app_env":"development"}
```

`docker compose up` brings up the full stack. Postgres becomes healthy, migrations
run, then the API starts. First build takes ~2 minutes (downloads base images and
installs packages); subsequent builds are fast thanks to layer caching.

## Quickstart (local development)

> Prerequisites: Python 3.12+, uv, Docker Desktop (for Postgres only)

```bash
git clone <repo-url> && cd ai-intel
uv sync --all-groups          # install deps + dev deps
cp .env.example .env          # set OPENAI_API_KEY and review defaults
make db-up                    # start only the Postgres container
make migrate                  # apply Alembic migrations
uv run uvicorn ai_intel.api.main:app --reload
# → http://localhost:8000/health
```

## Configuration

All configuration is loaded from environment variables (or a `.env` file) via
pydantic-settings. Copy `.env.example` to `.env` to get started.

| Variable | Required | Default | Description |
|---|---|---|---|
| `DATABASE_URL` | ✅ | — | asyncpg connection string (`postgresql+asyncpg://...`) |
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `POSTGRES_USER` | compose only | — | Postgres username (used by the `db` service) |
| `POSTGRES_PASSWORD` | compose only | — | Postgres password |
| `POSTGRES_DB` | compose only | — | Postgres database name |
| `GITHUB_TOKEN` | scraper | — | GitHub personal access token (read:public_repo) |
| `OPENAI_CHAT_MODEL` | — | `gpt-4o-mini` | Chat completion model |
| `OPENAI_EMBED_MODEL` | — | `text-embedding-3-small` | Embedding model (1536 dims) |
| `APP_ENV` | — | `development` | One of `development`, `staging`, `production` |
| `LOG_LEVEL` | — | `INFO` | One of `DEBUG`, `INFO`, `WARNING`, `ERROR` |

**Note on `DATABASE_URL`:** use `localhost` as the host when running locally
(`uv run alembic upgrade head`). The compose stack overrides this to `db` (the
service hostname) automatically — you don't need to edit it.

## Project Structure

```
ai-intel/
├── src/ai_intel/           Core package
│   ├── api/                FastAPI app and route definitions
│   ├── cli.py              argparse CLI — `ai-intel-scrape` entry point
│   ├── models/             SQLAlchemy ORM models (Item)
│   ├── sources/            Scrapers
│   │   ├── base.py         SourceItem dataclass + BaseSource ABC
│   │   ├── github.py       GitHub search API scraper
│   │   ├── repository.py   ItemRepository — upsert_many, most_recent_published_at
│   │   └── runner.py       run_source orchestration
│   ├── rag/                RAG pipeline — placeholder, Phase 7+
│   ├── agent/              Tool-calling agent — placeholder, Phase 8+
│   ├── mcp_server/         MCP server — placeholder, Phase 9+
│   ├── observability/      Dashboard backend — placeholder, Phase 10+
│   ├── config.py           pydantic-settings: loads .env, typed fields
│   └── db.py               Async engine, session factory, DeclarativeBase
├── alembic/                Alembic migrations (async env.py)
│   └── versions/           One migration per schema change
├── tests/                  pytest suite
├── .github/workflows/      CI (lint + tests on push)
├── Dockerfile              Multi-stage: builder (uv) → runtime (non-root)
├── docker-compose.yml      db + migrate + app
├── alembic.ini             Alembic config (URL set at runtime from .env)
├── pyproject.toml          Project metadata, dependencies, tool config
└── Makefile                Developer shortcuts (see below)
```

## Development

### Common commands

| Command | What it does |
|---|---|
| `make install` | Install all deps including dev (`uv sync --all-groups`) |
| `make dev` | Run FastAPI with hot-reload on port 8000 |
| `make test` | Run the full pytest suite |
| `make lint` | Check code with ruff |
| `make format` | Format with black + fix import order with ruff |
| `make up` | Start the full compose stack in the background |
| `make up-build` | Rebuild images and start the stack |
| `make down` | Stop and remove all containers (volume preserved) |
| `make logs` | Tail logs from all services |
| `make logs-app` | Tail logs from the app service only |
| `make ps` | Show service status and health |
| `make migrate` | Apply pending migrations locally (requires `make db-up`) |
| `make migrate-docker` | Apply pending migrations inside the compose network |
| `make clean` | Tear down stack + remove volumes and Python caches |

### Database migrations

Create a new migration after changing a model:

```bash
make revision m="add source_score column to items"
# → generates alembic/versions/<rev>_add_source_score_column_to_items.py
```

Review the generated file before applying. If you add a PostgreSQL extension or
index with special options (e.g. GIN, HNSW), add it manually — autogenerate won't
detect extensions and may not capture all index options.

Apply migrations:

```bash
make migrate          # local: runs against localhost:5432
make migrate-docker   # inside compose: runs against the db service
```

Roll back one step: `make migrate-down`.

### Tests

```bash
make test             # equivalent to: uv run pytest -v
```

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` — async test functions and
fixtures work without explicit marks. The `patch_settings` fixture in
`tests/conftest.py` injects dummy env vars and clears the `lru_cache` so tests
never depend on a real `.env` file.

No live database is required to run the current test suite.

## Scrapers

### GitHub

Searches the GitHub API for repositories tagged with AI-related topics
(`artificial-intelligence`, `machine-learning`, `llm`, `generative-ai`).
Paginates up to 500 results per run, upserts into the `items` table, and
automatically resumes from the last seen `pushed_at` on the next run.

**Setup:** add your token to `.env`:

```
GITHUB_TOKEN=ghp_...
```

**Run:**

```bash
make scrape-github                  # resume from last seen date
make scrape-github since=2026-01-01 # fetch everything pushed after this date
```

Or via the CLI directly:

```bash
uv run ai-intel-scrape --source github --since 2026-01-01
```

## Roadmap

In priority order:

- **Source scrapers** — daily ingestion from GitHub trending, Reddit r/MachineLearning
  and r/artificial, Product Hunt, Y Combinator new companies, Hacker News
- **Embedding pipeline** — batch embed items with `text-embedding-3-small`, store in
  the `embedding` vector column, add HNSW index
- **RAG-powered chat** — semantic search endpoint, context assembly, cited answers
- **Agent with tool calling** — structured tools: keyword search, trending items,
  source comparison, thread summarisation
- **MCP server** — expose the agent tools via Anthropic's MCP protocol
- **Observability dashboard** — per-query metrics: retrieval scores, token costs,
  latency breakdown, full reasoning chain
- **Automated daily digest** — scheduled summary of the top items across all sources

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
