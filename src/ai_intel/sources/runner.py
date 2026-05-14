import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ai_intel.sources.base import BaseSource
from ai_intel.sources.repository import ItemRepository

logger = logging.getLogger(__name__)


async def run_source(
    source: BaseSource,
    session: AsyncSession,
    *,
    since: datetime | None = None,
) -> int:
    repo = ItemRepository(session)
    if since is None:
        since = await repo.most_recent_published_at(source.source_name)
        if since:
            logger.info(
                "Resuming %s from last seen published_at=%s",
                source.source_name,
                since.isoformat(),
            )
    logger.info("Fetching from source=%s since=%s", source.source_name, since)
    items = await source.fetch(since=since)
    logger.info("Fetched %d items from %s", len(items), source.source_name)
    count = await repo.upsert_many(items)
    logger.info("Upserted %d rows for %s", count, source.source_name)
    return count
