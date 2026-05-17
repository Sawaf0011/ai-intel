from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ai_intel.models.item import Item
from ai_intel.sources.base import SourceItem


class ItemRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_many(self, items: Sequence[SourceItem]) -> int:
        if not items:
            return 0
        rows = [
            {
                "id": item.id,
                "source": item.source,
                "title": item.title,
                "url": item.url,
                "content": item.content,
                "author": item.author,
                "published_at": item.published_at,
                "metadata": item.metadata,
            }
            for item in items
        ]
        stmt = insert(Item.__table__).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "title": stmt.excluded.title,
                "url": stmt.excluded.url,
                "content": stmt.excluded.content,
                "author": stmt.excluded.author,
                "published_at": stmt.excluded.published_at,
                "metadata": stmt.excluded["metadata"],
                "fetched_at": text("now()"),
            },
        )
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount

    async def most_recent_published_at(self, source: str) -> datetime | None:
        result = await self._session.execute(
            select(Item.published_at)
            .where(Item.source == source)
            .order_by(Item.published_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
