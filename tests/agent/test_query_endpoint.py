"""Unit tests for the POST /query endpoint.

The IntelligenceAgent is mocked — no real OpenAI or DB calls.
"""

from unittest.mock import AsyncMock, patch

from ai_intel.agent.agent import AgentResponse

# ---------------------------------------------------------------------------
# Happy-path responses
# ---------------------------------------------------------------------------


async def test_query_returns_expected_shape(client):
    """POST /query with a valid question returns the full response schema."""
    mock_response = AgentResponse(
        answer="Here are the top AI agent frameworks...",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "agent"}}],
        sources=[{"id": "gh:1", "source": "github", "title": "AgentLib", "url": "u"}],
        iterations=2,
        hit_iteration_limit=False,
    )

    with patch(
        "ai_intel.api.routes.query.IntelligenceAgent.answer",
        AsyncMock(return_value=mock_response),
    ):
        response = await client.post("/query", json={"question": "What agent frameworks exist?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Here are the top AI agent frameworks..."
    assert len(body["sources"]) == 1
    assert len(body["tool_calls"]) == 1
    assert body["iterations"] == 2
    assert body["hit_iteration_limit"] is False


async def test_query_empty_sources_and_tool_calls(client):
    """A response with no tool calls is valid (model answered directly)."""
    mock_response = AgentResponse(
        answer="I don't have relevant data on that.",
        tool_calls=[],
        sources=[],
        iterations=1,
    )

    with patch(
        "ai_intel.api.routes.query.IntelligenceAgent.answer",
        AsyncMock(return_value=mock_response),
    ):
        response = await client.post("/query", json={"question": "A question"})

    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"] == []
    assert body["sources"] == []
    assert body["iterations"] == 1


# ---------------------------------------------------------------------------
# Validation errors — 422
# ---------------------------------------------------------------------------


async def test_query_missing_question_field_422(client):
    """Missing 'question' field must return 422."""
    response = await client.post("/query", json={})
    assert response.status_code == 422


async def test_query_empty_string_question_422(client):
    """Empty string question must return 422 (min_length=1)."""
    response = await client.post("/query", json={"question": ""})
    assert response.status_code == 422


async def test_query_wrong_content_type_422(client):
    """Non-JSON body must return 422."""
    response = await client.post(
        "/query",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Agent error → 500
# ---------------------------------------------------------------------------


async def test_query_agent_error_returns_500(client):
    """If the agent raises, the endpoint returns 500 with a clean message."""
    with patch(
        "ai_intel.api.routes.query.IntelligenceAgent.answer",
        AsyncMock(side_effect=RuntimeError("OpenAI is down")),
    ):
        response = await client.post("/query", json={"question": "Will this break?"})

    assert response.status_code == 500
    assert "Agent error" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /health still works
# ---------------------------------------------------------------------------


async def test_health_still_works(client):
    """/health must be unaffected by the new /query route."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
