"""Vector similarity search service using pgvector cosine distance.

pgvector's <=> operator returns cosine *distance* in [0, 2].
We convert to cosine *similarity* = 1 - distance so higher = more relevant.
The HNSW index built in Phase 7 (vector_cosine_ops) is used automatically
when the query is ordered by <=> on the embedding column.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_intel.embeddings.embedder import Embedder
from ai_intel.models.item import Item


@dataclass(slots=True)
class SearchResult:
    id: str
    source: str
    title: str
    url: str
    content: str | None
    metadata: dict
    similarity: float  # cosine similarity = 1 - cosine_distance; higher is better


class SearchService:
    """Semantic search over the items table via pgvector cosine distance."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def semantic_search(
        self,
        query: str,
        *,
        source: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return items ranked by cosine similarity to the query string.

        Embeds the query with Embedder, then queries the HNSW index via pgvector's
        <=> cosine-distance operator. Only considers rows where embedding IS NOT NULL.
        Converts distance to similarity: similarity = 1 - distance.

        Args:
            query: Natural-language query string.
            source: If provided, restrict results to items from this source.
            limit: Maximum number of results to return.
        """
        embedder = Embedder()
        vectors = await embedder.embed_texts([query])
        query_vector = vectors[0]

        dist_expr = Item.embedding.cosine_distance(query_vector).label("distance")

        stmt = (
            select(Item, dist_expr)
            .where(Item.embedding.is_not(None))
            .order_by(dist_expr)
            .limit(limit)
        )
        if source is not None:
            stmt = stmt.where(Item.source == source)

        result = await self._session.execute(stmt)
        rows = result.all()

        return [
            SearchResult(
                id=item.id,
                source=item.source,
                title=item.title,
                url=item.url,
                content=item.content,
                metadata=item.metadata_,
                similarity=float(1.0 - distance),
            )
            for item, distance in rows
        ]
