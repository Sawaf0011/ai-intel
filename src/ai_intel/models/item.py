from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ai_intel.db import Base


class Item(Base):
    __tablename__ = "items"
    __table_args__ = (
        # Composite B-tree index; DESC on published_at is set in the migration
        Index("idx_items_source_published", "source", "published_at"),
        # GIN index on the metadata JSONB column
        Index("idx_items_metadata", "metadata", postgresql_using="gin"),
    )

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    # Python attribute is metadata_ because 'metadata' is reserved on DeclarativeBase.
    # The actual DB column name is 'metadata' via the positional string arg.
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    fetched_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(1536), nullable=True)

    def __repr__(self) -> str:
        return f"<Item {self.source}:{self.id} {self.title[:40]}>"
