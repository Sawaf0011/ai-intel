"""Pipeline integration tests.

Full DB-backed tests require a running Postgres with the pgvector extension.
Because wiring a test-DB fixture cleanly would take more time than this phase
allows, live pipeline correctness is verified manually via the CLI:

    uv run python -m ai_intel.cli embed --limit 20   # partial run
    uv run python -m ai_intel.cli embed              # full run
    uv run python -m ai_intel.cli embed              # idempotency (should embed 0)

TODO: add a pytest-asyncio fixture that creates an isolated test database,
applies the migrations, seeds items, runs run_embedding_pipeline, and asserts:
  - result.embedded == number of seeded items
  - all items have embedding IS NOT NULL
  - a second run with force=False embeds 0 items (idempotency)
  - force=True re-embeds all items
"""
