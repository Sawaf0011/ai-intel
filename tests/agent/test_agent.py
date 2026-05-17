"""Unit tests for IntelligenceAgent.

All OpenAI API calls are mocked — no real API calls or DB access.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from ai_intel.agent.agent import (
    MAX_ITERATIONS,
    AgentResponse,
    IntelligenceAgent,
    _extract_sources,
)

# ---------------------------------------------------------------------------
# Helpers — build mock OpenAI response objects
# ---------------------------------------------------------------------------


def _make_tool_call(name: str, args: dict, call_id: str = "call_1") -> MagicMock:
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(args)
    return tc


def _make_assistant_msg(content: str | None = None, tool_calls=None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.model_dump.return_value = {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
    }
    return msg


def _make_response(msg: MagicMock) -> MagicMock:
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _patched_agent(create_side_effect) -> IntelligenceAgent:
    """Return an IntelligenceAgent whose OpenAI client is fully mocked."""
    with patch("ai_intel.agent.agent.openai.AsyncOpenAI"):
        agent = IntelligenceAgent()
    agent._client = MagicMock()
    agent._client.chat.completions.create = AsyncMock(side_effect=create_side_effect)
    return agent


# ---------------------------------------------------------------------------
# _extract_sources — pure function
# ---------------------------------------------------------------------------


def test_extract_sources_search_knowledge_base():
    result = [
        {"id": "gh:1", "source": "github", "title": "Repo", "url": "https://g.com", "similarity": 0.9},
    ]
    sources = _extract_sources("search_knowledge_base", result)
    assert sources == [{"id": "gh:1", "source": "github", "title": "Repo", "url": "https://g.com"}]


def test_extract_sources_get_item_details():
    result = {"id": "hn:42", "source": "hackernews", "title": "HN post", "url": "https://hn.com", "content": "x"}
    sources = _extract_sources("get_item_details", result)
    assert sources == [{"id": "hn:42", "source": "hackernews", "title": "HN post", "url": "https://hn.com"}]


def test_extract_sources_get_item_details_none():
    assert _extract_sources("get_item_details", None) == []


def test_extract_sources_compare_sources():
    result = {
        "github": [{"id": "gh:1", "source": "github", "title": "G", "url": "u1"}],
        "hackernews": [{"id": "hn:2", "source": "hackernews", "title": "H", "url": "u2"}],
        "ycombinator": [],
    }
    sources = _extract_sources("compare_sources", result)
    ids = {s["id"] for s in sources}
    assert ids == {"gh:1", "hn:2"}


def test_extract_sources_unknown_tool():
    assert _extract_sources("mystery_tool", {"anything": 1}) == []


# ---------------------------------------------------------------------------
# AgentResponse dataclass
# ---------------------------------------------------------------------------


def test_agent_response_defaults():
    r = AgentResponse(answer="ok", tool_calls=[], sources=[], iterations=1)
    assert r.hit_iteration_limit is False


# ---------------------------------------------------------------------------
# answer() — happy path: tool call then final answer
# ---------------------------------------------------------------------------


async def test_answer_tool_call_then_final_text():
    """Model calls search_knowledge_base once, then returns a final answer."""
    tool_call = _make_tool_call("search_knowledge_base", {"query": "agent frameworks"})
    first_msg = _make_assistant_msg(tool_calls=[tool_call])
    final_msg = _make_assistant_msg(content="Here are the top agent frameworks...")

    tool_result = [{"id": "gh:1", "source": "github", "title": "AgentLib", "url": "u"}]

    call_count = {"n": 0}

    async def create_side_effect(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _make_response(first_msg)
        return _make_response(final_msg)

    with patch(
        "ai_intel.retrieval.tools.search_knowledge_base",
        AsyncMock(return_value=tool_result),
    ):
        agent = _patched_agent(create_side_effect)
        result = await agent.answer("What agent frameworks exist?")

    assert result.answer == "Here are the top agent frameworks..."
    assert result.iterations == 2
    assert result.hit_iteration_limit is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "search_knowledge_base"
    assert result.tool_calls[0]["args"] == {"query": "agent frameworks"}


async def test_answer_immediate_final_text():
    """Model returns a text answer without any tool calls."""
    final_msg = _make_assistant_msg(content="I don't have data on that.")
    agent = _patched_agent(AsyncMock(return_value=_make_response(final_msg)))
    result = await agent.answer("What did Apple announce at WWDC?")

    assert result.answer == "I don't have data on that."
    assert result.iterations == 1
    assert result.tool_calls == []
    assert result.sources == []


# ---------------------------------------------------------------------------
# answer() — MAX_ITERATIONS enforcement
# ---------------------------------------------------------------------------


async def test_answer_stops_at_max_iterations():
    """Agent must stop and not loop forever even if model keeps calling tools."""
    tool_call = _make_tool_call("search_knowledge_base", {"query": "loop"})
    looping_msg = _make_assistant_msg(tool_calls=[tool_call])

    with patch(
        "ai_intel.retrieval.tools.search_knowledge_base",
        AsyncMock(return_value=[]),
    ):
        agent = _patched_agent(AsyncMock(return_value=_make_response(looping_msg)))
        result = await agent.answer("Keep looping")

    assert result.iterations == MAX_ITERATIONS
    assert result.hit_iteration_limit is True


# ---------------------------------------------------------------------------
# answer() — tool dispatch
# ---------------------------------------------------------------------------


async def test_answer_dispatches_get_trending():
    tool_call = _make_tool_call(
        "get_trending", {"source": "github", "timeframe_days": 30}
    )
    first_msg = _make_assistant_msg(tool_calls=[tool_call])
    final_msg = _make_assistant_msg(content="Trending repos: ...")

    trending_result = [{"id": "gh:tf", "source": "github", "title": "TF", "url": "u"}]
    call_count = {"n": 0}

    async def create_side_effect(**kwargs):
        call_count["n"] += 1
        return _make_response(first_msg if call_count["n"] == 1 else final_msg)

    with patch(
        "ai_intel.retrieval.tools.get_trending",
        AsyncMock(return_value=trending_result),
    ) as mock_trending:
        agent = _patched_agent(create_side_effect)
        result = await agent.answer("What's trending on GitHub?")

    mock_trending.assert_called_once_with(source="github", timeframe_days=30)
    assert "get_trending" in [tc["name"] for tc in result.tool_calls]


async def test_answer_dispatches_compare_sources():
    tool_call = _make_tool_call("compare_sources", {"query": "AI safety"})
    first_msg = _make_assistant_msg(tool_calls=[tool_call])
    final_msg = _make_assistant_msg(content="Comparison: ...")
    call_count = {"n": 0}

    async def create_side_effect(**kwargs):
        call_count["n"] += 1
        return _make_response(first_msg if call_count["n"] == 1 else final_msg)

    compare_result = {"github": [], "hackernews": [], "ycombinator": []}
    with patch(
        "ai_intel.retrieval.tools.compare_sources",
        AsyncMock(return_value=compare_result),
    ) as mock_compare:
        agent = _patched_agent(create_side_effect)
        await agent.answer("Compare AI safety discussion across sources")

    mock_compare.assert_called_once_with(query="AI safety")


async def test_answer_unknown_tool_returns_error_without_crashing():
    """An unrecognised tool name must not raise — it should return an error dict."""
    tool_call = _make_tool_call("nonexistent_tool", {"foo": "bar"})
    first_msg = _make_assistant_msg(tool_calls=[tool_call])
    final_msg = _make_assistant_msg(content="Sorry, something went wrong.")
    call_count = {"n": 0}

    async def create_side_effect(**kwargs):
        call_count["n"] += 1
        return _make_response(first_msg if call_count["n"] == 1 else final_msg)

    agent = _patched_agent(create_side_effect)
    result = await agent.answer("Trigger unknown tool")
    # Should not raise; should reach the final answer
    assert result.answer == "Sorry, something went wrong."


# ---------------------------------------------------------------------------
# answer() — source deduplication
# ---------------------------------------------------------------------------


async def test_answer_deduplicates_sources():
    """The same item returned by two different tool calls must appear once in sources."""
    item = {"id": "gh:1", "source": "github", "title": "Repo", "url": "u"}

    tc1 = _make_tool_call("search_knowledge_base", {"query": "q1"}, call_id="c1")
    tc2 = _make_tool_call("search_knowledge_base", {"query": "q2"}, call_id="c2")
    multi_tool_msg = _make_assistant_msg(tool_calls=[tc1, tc2])
    final_msg = _make_assistant_msg(content="Done.")

    call_count = {"n": 0}

    async def create_side_effect(**kwargs):
        call_count["n"] += 1
        return _make_response(multi_tool_msg if call_count["n"] == 1 else final_msg)

    with patch(
        "ai_intel.retrieval.tools.search_knowledge_base",
        AsyncMock(return_value=[item]),
    ):
        agent = _patched_agent(create_side_effect)
        result = await agent.answer("Dedup test")

    source_ids = [s["id"] for s in result.sources]
    assert source_ids.count("gh:1") == 1
