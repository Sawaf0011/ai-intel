"""Agent tool functions — deterministic retrieval primitives over the knowledge base.

Each function is a plain async callable with typed parameters and a JSON-serializable
return value. They are designed to be called directly (no LLM required) and are
registered as OpenAI tool schemas in the agent layer.

All tools open and close their own DB sessions via session_factory, so they are
independently callable without external session management.
"""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import Integer, func, nulls_last, select

from ai_intel.db import session_factory
from ai_intel.models.item import Item
from ai_intel.retrieval.search import SearchResult, SearchService

logger = logging.getLogger(__name__)

_VALID_SOURCES = frozenset({"github", "hackernews", "ycombinator"})
_CONTENT_PREVIEW_LEN = 300


def _truncate(text: str | None, max_len: int = _CONTENT_PREVIEW_LEN) -> str | None:
    """Truncate text to max_len characters; returns None unchanged."""
    if text is None:
        return None
    return text[:max_len] if len(text) > max_len else text


def _result_to_dict(result: SearchResult) -> dict:
    """Convert a SearchResult to a compact, JSON-serializable dict for the agent."""
    return {
        "id": result.id,
        "source": result.source,
        "title": result.title,
        "url": result.url,
        "content": _truncate(result.content),
        "similarity": round(result.similarity, 4),
    }


async def search_knowledge_base(
    query: str,
    source: str | None = None,
    limit: int = 8,
) -> list[dict]:
    """Search the AI startup knowledge base by semantic similarity.

    Returns items most similar to the query from GitHub repositories, Hacker News
    stories, and Y Combinator companies. Optionally filter to a single source.

    Use this tool first for any question about specific technologies, companies,
    repositories, or concepts. The results are ranked by semantic relevance.

    Args:
        query: Natural-language search query (e.g. "open source LLM frameworks").
        source: Filter to one source — "github", "hackernews", or "ycombinator".
                Omit (None) to search all sources.
        limit: Maximum number of results to return (default 8, recommended max 20).

    Returns:
        List of matching items, each with: id, source, title, url, content
        (first 300 chars), similarity score (0-1, higher is more relevant).

    Raises:
        ValueError: If source is not a valid source name.
    """
    if source is not None and source not in _VALID_SOURCES:
        raise ValueError(
            f"Invalid source {source!r}. Must be one of: {sorted(_VALID_SOURCES)}"
        )

    async with session_factory() as session:
        service = SearchService(session)
        results = await service.semantic_search(query, source=source, limit=limit)

    return [_result_to_dict(r) for r in results]


async def get_trending(
    source: str,
    timeframe_days: int = 7,
    limit: int = 10,
) -> list[dict]:
    """Return top items from one source ranked by a source-appropriate popularity metric.

    Ranking metric by source:
    - "github": ranked by stars (descending). Items with no stars field rank last.
    - "hackernews": ranked by score (descending). Items with no score rank last.
    - "ycombinator": ranked by launch date (most recently launched first).
      No engagement metric is available for YC companies in this dataset.

    This is a pure SQL query — no embeddings involved.

    Args:
        source: One of "github", "hackernews", "ycombinator".
        timeframe_days: Only include items published in the last N days (default 7).
        limit: Maximum number of items to return (default 10).

    Returns:
        List of item dicts with id, source, title, url, content snippet,
        published_at, and a subset of metadata relevant to the source.

    Raises:
        ValueError: If source is not a valid source name.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(
            f"Invalid source {source!r}. Must be one of: {sorted(_VALID_SOURCES)}"
        )

    cutoff = datetime.now(UTC) - timedelta(days=timeframe_days)

    async with session_factory() as session:
        stmt = select(Item).where(Item.source == source).where(
            Item.published_at >= cutoff
        )

        if source == "github":
            order_expr = func.coalesce(
                Item.metadata_["stars"].astext.cast(Integer), 0
            ).desc()
        elif source == "hackernews":
            order_expr = func.coalesce(
                Item.metadata_["score"].astext.cast(Integer), 0
            ).desc()
        else:
            # ycombinator — most recently launched; NULLs sort last
            order_expr = nulls_last(Item.published_at.desc())

        stmt = stmt.order_by(order_expr).limit(limit)
        result = await session.execute(stmt)
        items = result.scalars().all()

    _metadata_keys = {
        "github": ("stars", "forks", "language", "topics"),
        "hackernews": ("score", "comments", "hn_type"),
        "ycombinator": ("batch", "one_liner", "status", "team_size"),
    }
    keep = _metadata_keys[source]

    return [
        {
            "id": item.id,
            "source": item.source,
            "title": item.title,
            "url": item.url,
            "content": _truncate(item.content),
            "published_at": (
                item.published_at.isoformat() if item.published_at else None
            ),
            "metadata": {k: v for k, v in item.metadata_.items() if k in keep},
        }
        for item in items
    ]


async def get_item_details(item_id: str) -> dict | None:
    """Return the full record for a single item by its unique ID.

    Use this to deep-dive into an item returned by search_knowledge_base or
    get_trending — it includes the complete content and all metadata fields.

    Args:
        item_id: The item's unique ID (e.g. "github:user/repo", "hn:12345",
                 "yc:company-slug"). Obtain from search_knowledge_base results.

    Returns:
        Full item dict with complete content and all metadata, or None if not found.
    """
    async with session_factory() as session:
        result = await session.execute(select(Item).where(Item.id == item_id))
        item = result.scalar_one_or_none()

    if item is None:
        return None

    return {
        "id": item.id,
        "source": item.source,
        "title": item.title,
        "url": item.url,
        "content": item.content,
        "author": item.author,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "metadata": item.metadata_,
    }


async def compare_sources(query: str, limit_per_source: int = 4) -> dict:
    """Search the knowledge base for the same query across all three sources.

    Useful for comparing how different ecosystems discuss a topic — e.g. how GitHub
    projects, HN discussions, and YC companies each approach "AI safety".

    Runs semantic_search once per source (github, hackernews, ycombinator) and
    returns results grouped by source name.

    Args:
        query: Natural-language query (e.g. "AI safety", "LLM fine-tuning").
        limit_per_source: Number of results per source (default 4).

    Returns:
        Dict keyed by source name, each value a list of matching items.
        Example: {"github": [...], "hackernews": [...], "ycombinator": [...]}
    """
    results: dict[str, list[dict]] = {}
    async with session_factory() as session:
        service = SearchService(session)
        for src in sorted(_VALID_SOURCES):
            source_results = await service.semantic_search(
                query, source=src, limit=limit_per_source
            )
            results[src] = [_result_to_dict(r) for r in source_results]
    return results
