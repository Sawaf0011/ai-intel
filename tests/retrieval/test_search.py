"""Unit tests for SearchService and SearchResult.

DB-dependent tests mock the AsyncSession — no real Postgres connection required.
The Embedder is mocked to return a fixed fake vector — no real OpenAI calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_intel.retrieval.search import SearchResult, SearchService

_FAKE_VECTOR = [0.1] * 1536


def _make_mock_item(
    *,
    item_id: str = "gh:repo",
    source: str = "github",
    title: str = "Test Repo",
    url: str = "https://github.com/test/repo",
    content: str | None = "A great AI tool",
    metadata: dict | None = None,
):
    """Build a MagicMock that looks like an Item ORM instance."""
    item = MagicMock()
    item.id = item_id
    item.source = source
    item.title = title
    item.url = url
    item.content = content
    item.metadata_ = metadata if metadata is not None else {}
    return item


def _make_mock_session(rows: list[tuple]) -> AsyncMock:
    """Return a mock AsyncSession whose execute() yields the given rows."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = rows
    mock_session.execute.return_value = mock_result
    return mock_session


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


def test_search_result_can_be_instantiated():
    sr = SearchResult(
        id="gh:repo",
        source="github",
        title="My Repo",
        url="https://example.com",
        content="stuff",
        metadata={"stars": 100},
        similarity=0.92,
    )
    assert sr.id == "gh:repo"
    assert sr.similarity == 0.92


# ---------------------------------------------------------------------------
# Distance → similarity conversion
# ---------------------------------------------------------------------------


async def test_similarity_is_one_minus_distance():
    """similarity = 1 - cosine_distance; verify the arithmetic."""
    item = _make_mock_item()
    distance = 0.15
    mock_session = _make_mock_session([(item, distance)])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("test")

    assert len(results) == 1
    assert results[0].similarity == pytest.approx(1.0 - distance)


async def test_similarity_perfect_match():
    """Distance = 0 should yield similarity = 1.0."""
    item = _make_mock_item()
    mock_session = _make_mock_session([(item, 0.0)])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("test")

    assert results[0].similarity == pytest.approx(1.0)


async def test_similarity_worst_case():
    """Distance = 1.0 (opposite direction) → similarity = 0.0."""
    item = _make_mock_item()
    mock_session = _make_mock_session([(item, 1.0)])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("test")

    assert results[0].similarity == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# semantic_search — result mapping
# ---------------------------------------------------------------------------


async def test_semantic_search_maps_item_fields():
    """All fields from the Item ORM object must appear in the SearchResult."""
    item = _make_mock_item(
        item_id="yc:acme",
        source="ycombinator",
        title="Acme AI",
        url="https://acme.ai",
        content="AI-first company",
        metadata={"batch": "W24"},
    )
    mock_session = _make_mock_session([(item, 0.2)])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("AI company")

    r = results[0]
    assert r.id == "yc:acme"
    assert r.source == "ycombinator"
    assert r.title == "Acme AI"
    assert r.url == "https://acme.ai"
    assert r.content == "AI-first company"
    assert r.metadata == {"batch": "W24"}
    assert r.similarity == pytest.approx(0.8)


async def test_semantic_search_returns_multiple_results_in_order():
    """Results must come back in ascending distance (descending similarity) order."""
    items_and_distances = [
        (_make_mock_item(item_id=f"gh:{i}", title=f"Repo {i}"), float(i) * 0.1)
        for i in range(4)
    ]
    mock_session = _make_mock_session(items_and_distances)

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("query", limit=4)

    # similarity should be descending (best first)
    similarities = [r.similarity for r in results]
    assert similarities == sorted(similarities, reverse=True)


async def test_semantic_search_empty_db_returns_empty_list():
    mock_session = _make_mock_session([])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        results = await service.semantic_search("nothing here")

    assert results == []


async def test_semantic_search_calls_embed_texts_with_query():
    """The query string must be passed to embed_texts as a one-element list."""
    item = _make_mock_item()
    mock_session = _make_mock_session([(item, 0.1)])

    with patch("ai_intel.retrieval.search.Embedder") as MockEmbedder:
        mock_emb = AsyncMock()
        mock_emb.embed_texts = AsyncMock(return_value=[_FAKE_VECTOR])
        MockEmbedder.return_value = mock_emb

        service = SearchService(mock_session)
        await service.semantic_search("agent frameworks")

        mock_emb.embed_texts.assert_called_once_with(["agent frameworks"])
