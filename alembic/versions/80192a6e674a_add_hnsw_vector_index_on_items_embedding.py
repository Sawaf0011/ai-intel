"""add hnsw vector index on items embedding

Revision ID: 80192a6e674a
Revises: 5378ccff75d9
Create Date: 2026-05-17 14:58:53.282037

HNSW chosen over IVFFlat because:
  - Better recall/speed tradeoff for read-heavy RAG workloads
  - No training step required (IVFFlat needs CLUSTER before building)
  - Builds correctly even with a partially-populated table
  - Available in pgvector >= 0.5.0 (the pgvector/pgvector:pg16 image ships 0.8+)

vector_cosine_ops matches OpenAI embedding geometry; Phase 9 queries will
use the <=> cosine-distance operator.

Autogenerate is NOT used — Alembic does not handle pgvector index DDL.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "80192a6e674a"
down_revision: str | Sequence[str] | None = "5378ccff75d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create HNSW index on items.embedding using cosine distance."""
    op.execute(
        "CREATE INDEX idx_items_embedding_hnsw "
        "ON items USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Drop the HNSW vector index."""
    op.execute("DROP INDEX IF EXISTS idx_items_embedding_hnsw")
