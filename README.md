# AI Startup Intelligence & Observability Platform

A RAG + agent system that monitors the AI startup ecosystem across GitHub, Reddit,
Product Hunt, Y Combinator, and Hacker News. It exposes a semantic search chatbot,
an MCP server wrapping the same tools, and an observability dashboard tracking
retrieval scores, token costs, latency, and reasoning paths per query.

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Package manager | uv |
| Web framework | FastAPI |
| Database | Postgres 16 + pgvector |
| ORM | SQLAlchemy 2.x (async) + Alembic |
| Validation | Pydantic v2 |
| LLM SDK | OpenAI Python SDK |
| MCP SDK | `mcp` (Anthropic official) |
| Containers | Docker + docker-compose |
| Linting | ruff + black |
| Testing | pytest + pytest-asyncio |

## Getting Started

> Prerequisites: Python 3.11+, uv, Docker Desktop

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd ai-intel

# 2. Install dependencies
make install

# 3. Copy env file and fill in secrets
cp .env.example .env

# 4. Start Postgres
make db-up

# 5. Apply migrations and run the API
make migrate && make dev
```

The API will be available at `http://localhost:8000`. Check `http://localhost:8000/health`.

### Running the API without `make`

```bash
cp .env.example .env
# fill in OPENAI_API_KEY at minimum
uv run uvicorn ai_intel.api.main:app --reload
# visit http://localhost:8000/health
```

## Project Layout

```
src/ai_intel/       Core package
  api/              FastAPI app and routes
  models/           SQLAlchemy ORM models
  sources/          Scrapers (future)
  rag/              RAG pipeline (future)
  agent/            Tool-calling agent (future)
  mcp_server/       MCP server (future)
  observability/    Dashboard backend (future)
alembic/            Database migrations
tests/              pytest suite
scripts/            Dev convenience scripts
```

## Architecture Overview

_To be expanded as the system grows._

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
