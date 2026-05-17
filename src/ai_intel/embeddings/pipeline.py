"""Embedding pipeline: fetch un-embedded items, call OpenAI, persist vectors."""

import logging
from dataclasses import dataclass

from ai_intel.db import session_factory
from ai_intel.embeddings.embedder import Embedder, build_embedding_text
from ai_intel.embeddings.repository import EmbeddingRepository

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    embedded: int
    total_seen: int


async def run_embedding_pipeline(
    *, force: bool = False, limit: int | None = None
) -> EmbeddingResult:
    """Embed items that don't yet have an embedding (or all items if force=True).

    Commits after each batch of 100 so a crash mid-run keeps already-embedded
    items. Running again with force=False after a partial run is safe.
    """
    embedder = Embedder()
    embedded = 0

    async with session_factory() as session:
        repo = EmbeddingRepository(session)

        items = (
            await repo.all_items(limit=limit)
            if force
            else await repo.items_without_embedding(limit=limit)
        )
        total_seen = len(items)

        if total_seen == 0:
            logger.info("Embedding pipeline: no items to embed")
            return EmbeddingResult(embedded=0, total_seen=0)

        for start in range(0, total_seen, Embedder.BATCH_SIZE):
            batch_items = items[start : start + Embedder.BATCH_SIZE]

            # Build texts and track item IDs explicitly — never rely on
            # loose list alignment between two separate data structures.
            item_ids = [item.id for item in batch_items]
            texts = [build_embedding_text(item) for item in batch_items]

            vectors = await embedder.embed_texts(texts)

            # Map id → vector and write to DB.
            updates = dict(zip(item_ids, vectors))
            count = await repo.update_embeddings(updates)
            await session.commit()
            embedded += count

    logger.info(
        "Embedding pipeline complete: embedded=%d total_seen=%d", embedded, total_seen
    )
    return EmbeddingResult(embedded=embedded, total_seen=total_seen)
