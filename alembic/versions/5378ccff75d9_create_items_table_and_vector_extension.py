"""create items table and vector extension

Revision ID: 5378ccff75d9
Revises:
Create Date: 2026-05-14 16:55:32.043935

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5378ccff75d9"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Manually added: autogenerate does not detect PostgreSQL extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "items",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("source", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=True),
        sa.Column("author", sa.Text, nullable=True),
        sa.Column("published_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "fetched_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("embedding", Vector(1536), nullable=True),
    )

    # Composite B-tree index; published_at is DESC to match query patterns
    op.create_index(
        "idx_items_source_published",
        "items",
        ["source", sa.text("published_at DESC")],
    )

    # GIN index for efficient JSONB containment queries (@>, ?)
    op.create_index(
        "idx_items_metadata",
        "items",
        ["metadata"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("items")
    op.execute("DROP EXTENSION IF EXISTS vector")
