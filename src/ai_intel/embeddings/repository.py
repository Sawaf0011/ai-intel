"""EmbeddingRepository — data access for the embedding pipeline.

Kept separate from ItemRepository (sources/repository.py) because scraping
and embedding are distinct concerns. ItemRepository owns upsert logic; this
class owns embedding reads and bulk-updates.

None of the write methods commit. The pipeline controls transaction boundaries
(commit-per-batch) so a crash mid-run keeps already-embedded items.
"""

from collections.abc import Sequence

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ai_intel.models.item import Item


class EmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def items_without_embedding(self, limit: int | None = None) -> Sequence[Item]:
        """Items where embedding IS NULL, newest fetched_at first."""
        stmt = (
            select(Item)
            .where(Item.embedding.is_(None))
            .order_by(Item.fetched_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def all_items(self, limit: int | None = None) -> Sequence[Item]:
        """All items, newest fetched_at first (used for --force re-embed)."""
        stmt = select(Item).order_by(Item.fetched_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def count_without_embedding(self) -> int:
        """Count items still missing an embedding."""
        result = await self._session.execute(
            select(func.count()).select_from(Item).where(Item.embedding.is_(None))
        )
        return result.scalar_one()

    async def update_embeddings(self, updates: dict[str, list[float]]) -> int:
        """Bulk-update the embedding column for a mapping of item_id → vector.

        Issues one UPDATE per item. At batch size 100 this is fast enough;
        the alternative (executemany with bindparam) has driver-level
        complexity with pgvector types that isn't worth the added fragility.

        Does NOT commit — caller controls transaction boundaries.
        Returns number of rows actually updated.
        """
        if not updates:
            return 0
        count = 0
        for item_id, vector in updates.items():
            result = await self._session.execute(
                update(Item).where(Item.id == item_id).values(embedding=vector)
            )
            count += result.rowcount
        return count
