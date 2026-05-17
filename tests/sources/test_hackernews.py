"""Unit tests for HackerNewsSource.

All HTTP calls are mocked with respx — no real HN API calls.
"""

from datetime import UTC, datetime

import httpx
import respx

from ai_intel.sources.hackernews import HackerNewsSource, _is_ai_related

_BASE = "https://hacker-news.firebaseio.com/v0"

# ---------------------------------------------------------------------------
# Fixtures — reusable raw story dicts
# ---------------------------------------------------------------------------

_AI_STORY = {
    "id": 100,
    "type": "story",
    "title": "OpenAI releases GPT-5",
    "url": "https://example.com/gpt5",
    "by": "user1",
    "score": 500,
    "time": 1700000000,
    "descendants": 120,
}

_NON_AI_STORY = {
    "id": 101,
    "type": "story",
    "title": "The best pasta recipes of 2026",
    "url": "https://example.com/pasta",
    "by": "chef",
    "score": 80,
    "time": 1700000100,
    "descendants": 20,
}

_SELF_POST = {
    "id": 102,
    "type": "story",
    "title": "Ask HN: Best LLM tools for developers?",
    "text": "What LLM tools are you using day to day?",
    "by": "devask",
    "score": 200,
    "time": 1700000200,
    "descendants": 75,
    # No 'url' field — self-post
}

_COMMENT = {
    "id": 103,
    "type": "comment",
    "by": "commenter",
    "text": "Great post about AI",
    "time": 1700000300,
    "parent": 100,
}

_DEAD_STORY = {
    "id": 104,
    "type": "story",
    "title": "Claude 4 benchmark results",
    "url": "https://example.com/claude4",
    "by": "user2",
    "score": 300,
    "time": 1700000400,
    "descendants": 50,
    "dead": True,
}

_OLD_AI_STORY = {
    "id": 105,
    "type": "story",
    "title": "Machine learning breakthrough from 2020",
    "url": "https://example.com/old",
    "by": "olduser",
    "score": 100,
    "time": 1577836800,  # 2020-01-01 00:00:00 UTC
    "descendants": 10,
}


def _mock_ids(top: list[int], best: list[int]) -> None:
    respx.get(f"{_BASE}/topstories.json").mock(
        return_value=httpx.Response(200, json=top)
    )
    respx.get(f"{_BASE}/beststories.json").mock(
        return_value=httpx.Response(200, json=best)
    )


def _mock_item(story: dict) -> None:
    respx.get(f"{_BASE}/item/{story['id']}.json").mock(
        return_value=httpx.Response(200, json=story)
    )


# ---------------------------------------------------------------------------
# _is_ai_related keyword matching
# ---------------------------------------------------------------------------


def test_is_ai_related_long_keyword():
    assert _is_ai_related("OpenAI announces new model")
    assert _is_ai_related("Machine learning breakthrough")
    assert _is_ai_related("Anthropic releases Claude")


def test_is_ai_related_short_keyword_word_boundary():
    # Short keywords must be whole words
    assert _is_ai_related("Best AI tools for 2026")
    assert _is_ai_related("New LLM beats GPT")
    assert not _is_ai_related("The detail in this painting")  # 'ai' inside 'detail'
    assert not _is_ai_related("It was raining all day")  # 'ai' inside 'raining'
    assert not _is_ai_related("Email newsletter tips")  # 'ai' inside 'email'


def test_is_ai_related_case_insensitive():
    assert _is_ai_related("gpt-4 performance review")
    assert _is_ai_related("NEURAL NETWORK architectures")
    assert _is_ai_related("Transformer models explained")


def test_is_ai_related_no_match():
    assert not _is_ai_related("Python web frameworks compared")
    assert not _is_ai_related("The best pasta recipes of 2026")
    assert not _is_ai_related("How to invest in index funds")


# ---------------------------------------------------------------------------
# fetch() — core behavior
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_returns_only_ai_stories():
    _mock_ids(top=[100, 101], best=[])
    _mock_item(_AI_STORY)
    _mock_item(_NON_AI_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 1
    assert items[0].id == "hn:100"
    assert items[0].title == "OpenAI releases GPT-5"


@respx.mock
async def test_fetch_drops_non_story_types():
    _mock_ids(top=[103], best=[])
    _mock_item(_COMMENT)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 0


@respx.mock
async def test_fetch_drops_dead_stories():
    _mock_ids(top=[104], best=[])
    _mock_item(_DEAD_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 0


@respx.mock
async def test_fetch_self_post_uses_hn_permalink():
    """A story with no 'url' field must get the HN permalink as its url."""
    _mock_ids(top=[102], best=[])
    _mock_item(_SELF_POST)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 1
    assert items[0].url == "https://news.ycombinator.com/item?id=102"
    assert items[0].content == "What LLM tools are you using day to day?"


@respx.mock
async def test_fetch_unix_epoch_time_converts_correctly():
    """time=1700000000 must produce timezone-aware datetime."""
    _mock_ids(top=[100], best=[])
    _mock_item(_AI_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    expected = datetime.fromtimestamp(1700000000, tz=UTC)
    assert items[0].published_at == expected
    assert items[0].published_at.tzinfo is not None


@respx.mock
async def test_fetch_metadata_fields():
    _mock_ids(top=[100], best=[])
    _mock_item(_AI_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    m = items[0].metadata
    assert m["score"] == 500
    assert m["comments"] == 120
    assert m["hn_type"] == "story"
    assert m["hn_url"] == "https://news.ycombinator.com/item?id=100"


@respx.mock
async def test_fetch_deduplicates_across_top_and_best():
    """ID 100 appearing in both topstories and beststories must produce 1 item."""
    _mock_ids(top=[100], best=[100])
    _mock_item(_AI_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 1


@respx.mock
async def test_fetch_since_drops_older_items():
    """Items published before `since` must be excluded."""
    _mock_ids(top=[100, 105], best=[])
    _mock_item(_AI_STORY)
    _mock_item(_OLD_AI_STORY)

    # since=2023-01-01: _AI_STORY (2023-11-14) passes; _OLD_AI_STORY (2020-01-01) is dropped
    since = datetime(2023, 1, 1, tzinfo=UTC)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch(since=since)

    ids = {item.id for item in items}
    assert "hn:100" in ids  # 2023-11-14 — after since
    assert "hn:105" not in ids  # 2020-01-01 — before since


@respx.mock
async def test_fetch_source_name():
    _mock_ids(top=[100], best=[])
    _mock_item(_AI_STORY)

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert items[0].source == "hackernews"


@respx.mock
async def test_fetch_empty_lists():
    _mock_ids(top=[], best=[])

    async with httpx.AsyncClient() as client:
        source = HackerNewsSource(http_client=client)
        items = await source.fetch()

    assert items == []
