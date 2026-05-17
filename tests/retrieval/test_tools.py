"""Unit tests for retrieval tool functions.

Tests cover pure-logic pieces (source validation, content truncation, result shape)
and mock the DB session + Embedder to avoid requiring a live Postgres or OpenAI.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_intel.retrieval.search import SearchResult
from ai_intel.retrieval.tools import (
    _CONTENT_PREVIEW_LEN,
    _truncate,
    compare_sources,
    get_item_details,
    get_trending,
    search_knowledge_base,
)

# ---------------------------------------------------------------------------
# _truncate — pure function
# ---------------------------------------------------------------------------


def test_truncate_none_returns_none():
    assert _truncate(None) is None


def test_truncate_short_text_unchanged():
    assert _truncate("hello") == "hello"


def test_truncate_at_limit():
    text = "x" * _CONTENT_PREVIEW_LEN
    assert _truncate(text) == text


def test_truncate_long_text_clipped():
    long = "a" * (_CONTENT_PREVIEW_LEN + 100)
    result = _truncate(long)
    assert len(result) == _CONTENT_PREVIEW_LEN
    assert result == "a" * _CONTENT_PREVIEW_LEN


def test_truncate_custom_max_len():
    assert _truncate("hello world", max_len=5) == "hello"


# ---------------------------------------------------------------------------
# search_knowledge_base — source validation
# ---------------------------------------------------------------------------


async def test_search_knowledge_base_invalid_source_raises():
    with pytest.raises(ValueError, match="Invalid source"):
        await search_knowledge_base("query", source="twitter")


async def test_search_knowledge_base_none_source_is_valid():
    """source=None (all sources) must not raise."""
    fake_result = SearchResult(
        id="gh:1", source="github", title="T", url="u", content="c",
        metadata={}, similarity=0.9,
    )
    with (
        patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
        patch("ai_intel.retrieval.tools.SearchService") as MockService,
    ):
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.semantic_search = AsyncMock(return_value=[fake_result])
        MockService.return_value = mock_svc

        result = await search_knowledge_base("open LLMs", source=None)

    assert len(result) == 1
    mock_svc.semantic_search.assert_called_once_with(
        "open LLMs", source=None, limit=8
    )


async def test_search_knowledge_base_valid_source_passes():
    """Each valid source name must not raise."""
    fake_result = SearchResult(
        id="hn:1", source="hackernews", title="T", url="u", content=None,
        metadata={}, similarity=0.7,
    )
    for valid_source in ("github", "hackernews", "ycombinator"):
        with (
            patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
            patch("ai_intel.retrieval.tools.SearchService") as MockService,
        ):
            mock_session = AsyncMock()
            mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_svc = AsyncMock()
            mock_svc.semantic_search = AsyncMock(return_value=[fake_result])
            MockService.return_value = mock_svc

            result = await search_knowledge_base("query", source=valid_source)
        assert isinstance(result, list)


async def test_search_knowledge_base_content_is_truncated():
    """Tool must truncate content to _CONTENT_PREVIEW_LEN characters."""
    long_content = "z" * (_CONTENT_PREVIEW_LEN + 200)
    fake_result = SearchResult(
        id="gh:1", source="github", title="T", url="u",
        content=long_content, metadata={}, similarity=0.8,
    )
    with (
        patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
        patch("ai_intel.retrieval.tools.SearchService") as MockService,
    ):
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.semantic_search = AsyncMock(return_value=[fake_result])
        MockService.return_value = mock_svc

        result = await search_knowledge_base("query")

    assert len(result[0]["content"]) == _CONTENT_PREVIEW_LEN


async def test_search_knowledge_base_result_shape():
    """Each result dict must contain the expected keys."""
    fake_result = SearchResult(
        id="yc:co", source="ycombinator", title="Co", url="https://co.com",
        content="short", metadata={}, similarity=0.95,
    )
    with (
        patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
        patch("ai_intel.retrieval.tools.SearchService") as MockService,
    ):
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.semantic_search = AsyncMock(return_value=[fake_result])
        MockService.return_value = mock_svc

        result = await search_knowledge_base("query")

    item = result[0]
    assert set(item.keys()) == {"id", "source", "title", "url", "content", "similarity"}
    assert item["similarity"] == pytest.approx(0.95, abs=1e-3)


# ---------------------------------------------------------------------------
# get_trending — source validation
# ---------------------------------------------------------------------------


async def test_get_trending_invalid_source_raises():
    with pytest.raises(ValueError, match="Invalid source"):
        await get_trending("reddit")


async def test_get_trending_returns_list():
    with patch("ai_intel.retrieval.tools.session_factory") as mock_sf:
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_trending("github", timeframe_days=30, limit=5)

    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# get_item_details — not found returns None
# ---------------------------------------------------------------------------


async def test_get_item_details_not_found_returns_none():
    with patch("ai_intel.retrieval.tools.session_factory") as mock_sf:
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_item_details("gh:does-not-exist")

    assert result is None


async def test_get_item_details_found_returns_dict():
    from datetime import UTC, datetime

    item = MagicMock()
    item.id = "gh:repo"
    item.source = "github"
    item.title = "Repo"
    item.url = "https://github.com/test/repo"
    item.content = "Full content here"
    item.author = "dev"
    item.published_at = datetime(2026, 1, 1, tzinfo=UTC)
    item.metadata_ = {"stars": 500}

    with patch("ai_intel.retrieval.tools.session_factory") as mock_sf:
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = item
        mock_session.execute = AsyncMock(return_value=mock_result)

        result = await get_item_details("gh:repo")

    assert result is not None
    assert result["id"] == "gh:repo"
    assert result["content"] == "Full content here"
    assert result["metadata"] == {"stars": 500}
    assert "T" in result["published_at"]  # ISO format contains date separator


# ---------------------------------------------------------------------------
# compare_sources — structure
# ---------------------------------------------------------------------------


async def test_compare_sources_returns_all_three_sources():
    """compare_sources must return a dict with all three source keys."""
    fake_result = SearchResult(
        id="gh:1", source="github", title="T", url="u", content=None,
        metadata={}, similarity=0.8,
    )
    with (
        patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
        patch("ai_intel.retrieval.tools.SearchService") as MockService,
    ):
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.semantic_search = AsyncMock(return_value=[fake_result])
        MockService.return_value = mock_svc

        result = await compare_sources("AI safety")

    assert set(result.keys()) == {"github", "hackernews", "ycombinator"}
    # semantic_search must be called once per source
    assert mock_svc.semantic_search.call_count == 3


async def test_compare_sources_each_value_is_list():
    with (
        patch("ai_intel.retrieval.tools.session_factory") as mock_sf,
        patch("ai_intel.retrieval.tools.SearchService") as MockService,
    ):
        mock_session = AsyncMock()
        mock_sf.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_sf.return_value.__aexit__ = AsyncMock(return_value=False)

        mock_svc = AsyncMock()
        mock_svc.semantic_search = AsyncMock(return_value=[])
        MockService.return_value = mock_svc

        result = await compare_sources("LLM fine-tuning", limit_per_source=3)

    for source, items in result.items():
        assert isinstance(items, list), f"Expected list for source {source!r}"
