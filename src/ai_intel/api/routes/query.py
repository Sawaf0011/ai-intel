"""POST /query — natural-language question answering over the knowledge base."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ai_intel.agent.agent import AgentResponse, IntelligenceAgent

logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    tool_calls: list[dict]
    iterations: int
    hit_iteration_limit: bool = False


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural-language question using the AI startup knowledge base.

    The agent uses semantic search, trending queries, and cross-source comparison
    to ground its answer in real data. All cited sources are returned.
    """
    try:
        agent = IntelligenceAgent()
        result: AgentResponse = await agent.answer(request.question)
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            tool_calls=result.tool_calls,
            iterations=result.iterations,
            hit_iteration_limit=result.hit_iteration_limit,
        )
    except Exception as exc:
        logger.exception("Agent error for question %r", request.question)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
