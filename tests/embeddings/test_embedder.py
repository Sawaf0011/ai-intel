"""Unit tests for Embedder and build_embedding_text.

All tests mock AsyncOpenAI — no real API calls are made.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_intel.embeddings.embedder import (  # noqa: E402
    BATCH_SIZE,
    EMBEDDING_DIM,
    Embedder,
    build_embedding_text,
)
from ai_intel.models.item import Item

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    *,
    title: str = "Test Repo",
    content: str | None = "A great AI tool",
    source: str = "github",
    metadata: dict | None = None,
) -> Item:
    item = Item()
    item.id = "github:1"
    item.source = source
    item.title = title
    item.url = "https://github.com/test/repo"
    item.content = content
    item.author = "testuser"
    item.published_at = datetime(2026, 1, 1, tzinfo=UTC)
    item.fetched_at = datetime(2026, 1, 2, tzinfo=UTC)
    item.metadata_ = metadata if metadata is not None else {}
    item.embedding = None
    return item


def _fake_vector(index: int = 0) -> list[float]:
    """Return a correctly-sized fake embedding vector."""
    base = [0.0] * EMBEDDING_DIM
    base[index % EMBEDDING_DIM] = 1.0
    return base


def _make_mock_response(texts: list[str], *, start_index: int = 0) -> MagicMock:
    """Build a fake CreateEmbeddingResponse for the given texts."""
    response = MagicMock()
    response.data = [
        MagicMock(index=start_index + i, embedding=_fake_vector(start_index + i))
        for i in range(len(texts))
    ]
    return response


# ---------------------------------------------------------------------------
# build_embedding_text
# ---------------------------------------------------------------------------


def test_build_embedding_text_basic():
    item = _make_item(
        title="My Repo",
        content="Does cool things",
        source="github",
        metadata={"language": "Python", "topics": ["llm", "agents"], "stars": 500},
    )
    text = build_embedding_text(item)

    assert "My Repo" in text
    assert "Does cool things" in text
    assert "Source: github" in text
    assert "Language: Python" in text
    assert "Topics: llm, agents" in text
    assert "Stars: 500" in text


def test_build_embedding_text_no_content():
    item = _make_item(content=None, metadata={})
    text = build_embedding_text(item)
    assert "Test Repo" in text
    assert "Source: github" in text
    # No crash and no None literal in the output
    assert "None" not in text


def test_build_embedding_text_no_metadata():
    item = _make_item(metadata={})
    text = build_embedding_text(item)
    # No metadata fields — still well-formed
    assert "Test Repo" in text
    assert "Source: github" in text


def test_build_embedding_text_topics_capped_at_5():
    many_topics = [f"topic{i}" for i in range(10)]
    item = _make_item(metadata={"topics": many_topics})
    text = build_embedding_text(item)
    # Only the first 5 topics should appear
    assert "topic0" in text
    assert "topic4" in text
    assert "topic5" not in text


# ---------------------------------------------------------------------------
# Embedder.embed_texts — batching and order
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder_with_mock():
    """Return an Embedder whose _embed_batch is replaced with a controlled mock."""
    with patch("ai_intel.embeddings.embedder.openai.AsyncOpenAI"):
        emb = Embedder()

    call_count = {"n": 0}

    async def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        start = call_count["n"] * BATCH_SIZE
        call_count["n"] += 1
        return [_fake_vector(start + i) for i in range(len(texts))]

    emb._embed_batch = fake_embed_batch
    return emb, call_count


async def test_embed_texts_empty_returns_empty_no_api_call():
    with patch("ai_intel.embeddings.embedder.openai.AsyncOpenAI"):
        emb = Embedder()
    called = []
    emb._embed_batch = AsyncMock(side_effect=lambda t: called.append(t) or [])

    result = await emb.embed_texts([])

    assert result == []
    assert len(called) == 0


async def test_embed_texts_single_batch_one_api_call(embedder_with_mock):
    emb, call_count = embedder_with_mock
    texts = [f"text {i}" for i in range(50)]

    result = await emb.embed_texts(texts)

    assert len(result) == 50
    assert call_count["n"] == 1


async def test_embed_texts_batches_correctly_250_texts(embedder_with_mock):
    """250 texts → 3 batches (100, 100, 50)."""
    emb, call_count = embedder_with_mock
    texts = [f"text {i}" for i in range(250)]

    result = await emb.embed_texts(texts)

    assert call_count["n"] == 3
    assert len(result) == 250


async def test_embed_texts_order_preserved_across_batches():
    """Vectors at position i must correspond to text at position i."""
    with patch("ai_intel.embeddings.embedder.openai.AsyncOpenAI"):
        emb = Embedder()

    # Each batch returns vectors tagged with their position via the 0th element.
    call_log: list[int] = []

    async def fake_embed_batch(texts: list[str]) -> list[list[float]]:
        start = len(call_log) * BATCH_SIZE
        call_log.append(start)
        return [
            [float(start + i)] + [0.0] * (EMBEDDING_DIM - 1) for i in range(len(texts))
        ]

    emb._embed_batch = fake_embed_batch

    texts = [f"t{i}" for i in range(250)]
    result = await emb.embed_texts(texts)

    assert len(result) == 250
    for i, vec in enumerate(result):
        assert vec[0] == float(i), (
            f"Position {i} has wrong vector (first element {vec[0]})"
        )


# ---------------------------------------------------------------------------
# Embedder._embed_batch — dimension assertion
# ---------------------------------------------------------------------------


async def test_embed_batch_raises_on_wrong_dimension():
    """A misconfigured model returning wrong-size vectors must raise ValueError."""
    with patch("ai_intel.embeddings.embedder.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        bad_response = MagicMock()
        bad_response.data = [MagicMock(index=0, embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create = AsyncMock(return_value=bad_response)

        emb = Embedder()
        with pytest.raises(ValueError, match="Expected 1536-dim"):
            await emb._embed_batch(["test text"])


async def test_embed_batch_sorts_response_by_index():
    """If the API returns data out-of-order, _embed_batch must sort by index."""
    with patch("ai_intel.embeddings.embedder.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        vec_a = _fake_vector(0)
        vec_b = _fake_vector(1)
        # Return in reverse order (index 1 before index 0)
        response = MagicMock()
        response.data = [
            MagicMock(index=1, embedding=vec_b),
            MagicMock(index=0, embedding=vec_a),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=response)

        emb = Embedder()
        result = await emb._embed_batch(["first", "second"])

    assert result[0] == vec_a
    assert result[1] == vec_b
