"""Tool-calling intelligence agent over the AI startup knowledge base.

Uses the OpenAI chat completions API with tool calling. The agent loop:
  1. Sends the system prompt + user question with the four Part A tool schemas.
  2. If the model returns tool calls, dispatches each to the corresponding
     async function, appends results to the message list, and loops.
  3. If the model returns a final text answer, returns it.
  4. Stops after MAX_ITERATIONS to prevent runaway token spend.

All answers must be grounded in tool results — the system prompt enforces this.
Retrieved items are raw ecosystem signal (social posts, scraped data), not
verified facts, so the agent attributes claims to their source rather than
asserting them as truth.
"""

import json
import logging
from dataclasses import dataclass

import openai

import ai_intel.retrieval.tools as _tools
from ai_intel.config import get_settings

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 6

_SYSTEM_PROMPT = """You are an AI-ecosystem intelligence assistant. You have access \
to a knowledge base of items scraped from three sources:
- GitHub: AI/ML repositories (stars, forks, topics, descriptions)
- Hacker News: AI-related discussions and stories (scores, comments)
- Y Combinator: AI-tagged startup companies (batch, descriptions)

GROUNDING RULE: You MUST use the provided tools to answer every question. \
Do not answer from your own training knowledge. If the tools return nothing \
relevant to the question, say so clearly — do not fabricate or speculate.

ATTRIBUTION RULE: The knowledge base contains raw, unverified ecosystem signal — \
scraped social posts, repository descriptions, and startup blurbs. \
You MUST attribute claims to their source rather than asserting them as established \
fact. Use phrasing like:
  - "According to a Hacker News post titled '...'"
  - "A GitHub repository named '...' describes itself as..."
  - "A Y Combinator company called '...' states..."
Never present scraped content as verified truth.

CITATION RULE: Always cite the specific items you used (title + URL) at the end \
of your answer so the user can verify the sources.

Keep answers concise and focused on the question asked."""

# Hand-written tool schemas — explicit and easy to audit.
TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the AI startup knowledge base by semantic similarity. "
                "Returns items most similar to the query from GitHub repositories, "
                "Hacker News stories, and Y Combinator companies. "
                "Use this first for any question about specific technologies, "
                "companies, repositories, or concepts. Results are ranked by relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'open source LLM frameworks'.",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["github", "hackernews", "ycombinator"],
                        "description": (
                            "Filter to one source. Omit to search all sources."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 8, max 20).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trending",
            "description": (
                "Return top items from one source ranked by popularity. "
                "GitHub → ranked by stars. Hacker News → ranked by score. "
                "Y Combinator → ranked by launch date (most recent first). "
                "Use this to answer 'what's popular/trending in X' questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["github", "hackernews", "ycombinator"],
                        "description": "The source to query.",
                    },
                    "timeframe_days": {
                        "type": "integer",
                        "description": "Only include items from the last N days (default 7).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum items to return (default 10).",
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_details",
            "description": (
                "Return the full record for a single item by its unique ID. "
                "Use this to deep-dive into an item returned by other tools — "
                "it includes complete content and all metadata fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": (
                            "The item's unique ID from a previous tool result, "
                            "e.g. 'github:user/repo', 'hn:12345', 'yc:company-slug'."
                        ),
                    },
                },
                "required": ["item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_sources",
            "description": (
                "Search the knowledge base for the same query across all three sources "
                "(GitHub, Hacker News, Y Combinator) and return results grouped by source. "
                "Use this to compare how different ecosystems discuss a topic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query, e.g. 'AI safety'.",
                    },
                    "limit_per_source": {
                        "type": "integer",
                        "description": "Number of results per source (default 4).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]

_KNOWN_TOOLS = frozenset(
    {"search_knowledge_base", "get_trending", "get_item_details", "compare_sources"}
)


@dataclass(slots=True)
class AgentResponse:
    answer: str
    tool_calls: list[dict]  # {name, args} for each call made, in order
    sources: list[dict]  # {id, source, title, url} for cited items, deduplicated
    iterations: int
    hit_iteration_limit: bool = False


def _extract_sources(tool_name: str, result: object) -> list[dict]:
    """Pull {id, source, title, url} dicts out of a tool result."""
    keys = ("id", "source", "title", "url")

    if tool_name in ("search_knowledge_base", "get_trending") and isinstance(
        result, list
    ):
        return [{k: item[k] for k in keys if k in item} for item in result]

    if tool_name == "get_item_details" and isinstance(result, dict):
        return [{k: result[k] for k in keys if k in result}]

    if tool_name == "compare_sources" and isinstance(result, dict):
        out = []
        for items in result.values():
            for item in items:
                out.append({k: item[k] for k in keys if k in item})
        return out

    return []


class IntelligenceAgent:
    """OpenAI tool-calling agent over the AI startup knowledge base."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_chat_model

    async def answer(self, question: str) -> AgentResponse:
        """Answer a natural-language question using the knowledge-base tools.

        Runs the OpenAI tool-calling loop until the model produces a final text
        answer or MAX_ITERATIONS is reached. All tool calls are logged and their
        source items collected for citation.

        Args:
            question: The user's question in natural language.

        Returns:
            AgentResponse with the answer text, ordered tool-call trace,
            deduplicated source items, and iteration count.
        """
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        tool_calls_log: list[dict] = []
        seen_source_ids: set[str] = set()
        sources: list[dict] = []
        iterations = 0
        hit_limit = False

        while iterations < MAX_ITERATIONS:
            iterations += 1
            logger.info("Agent iteration %d/%d", iterations, MAX_ITERATIONS)

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
            )

            msg = response.choices[0].message

            # --- Final answer ---
            if not msg.tool_calls:
                return AgentResponse(
                    answer=msg.content or "",
                    tool_calls=tool_calls_log,
                    sources=sources,
                    iterations=iterations,
                    hit_iteration_limit=False,
                )

            # --- Tool calls: append assistant message, then execute each ---
            messages.append(msg.model_dump(exclude_unset=False))

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info("Tool call: %s(%s)", name, args)
                tool_calls_log.append({"name": name, "args": args})

                fn = getattr(_tools, name, None) if name in _KNOWN_TOOLS else None
                if fn is None:
                    result: object = {"error": f"Unknown tool: {name!r}"}
                else:
                    try:
                        result = await fn(**args)
                    except Exception as exc:
                        logger.warning("Tool %s failed: %s", name, exc)
                        result = {"error": str(exc)}

                # Collect source items for citation (guard against error dicts)
                for item in _extract_sources(name, result):
                    item_id = item.get("id")
                    if item_id and item_id not in seen_source_ids:
                        seen_source_ids.add(item_id)
                        sources.append(item)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

        # Iteration limit reached — return whatever the last content was
        hit_limit = True
        logger.warning("Agent hit MAX_ITERATIONS=%d for question: %r", MAX_ITERATIONS, question)
        last_content = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_content = m.get("content") or ""
                break
            if hasattr(m, "content") and m.content:
                last_content = m.content
                break

        return AgentResponse(
            answer=last_content or "I was unable to produce an answer within the iteration limit.",
            tool_calls=tool_calls_log,
            sources=sources,
            iterations=iterations,
            hit_iteration_limit=hit_limit,
        )
