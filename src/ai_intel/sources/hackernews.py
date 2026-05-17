"""Hacker News scraper using the official Firebase REST API.

No authentication required. API docs: https://github.com/HackerNews/API
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ai_intel.sources.base import BaseSource, SourceItem

logger = logging.getLogger(__name__)

_BASE_URL = "https://hacker-news.firebaseio.com/v0"
_HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"
_USER_AGENT = "ai-intel-scraper/0.1 (+https://github.com/placeholder/ai-intel)"

# AI-relevant keywords.  Short terms (ai, llm, gpt) use \b word-boundary matching
# to avoid false positives like "rain", "detail", "algorithm".
AI_KEYWORDS: list[str] = [
    "openai",
    "anthropic",
    "deepmind",
    "claude",
    "chatgpt",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "transformer",
    "diffusion model",
    "generative ai",
    "foundation model",
    "reinforcement learning",
    # Short-word patterns with explicit \b boundaries
]
_SHORT_KEYWORDS_RE = re.compile(
    r"\b(ai|ml|llm|gpt|rag|nlp|bert|gan)\b",
    re.IGNORECASE,
)
_LONG_KEYWORDS_RE = re.compile(
    "|".join(re.escape(kw) for kw in AI_KEYWORDS),
    re.IGNORECASE,
)


def _is_ai_related(title: str) -> bool:
    """Return True if the title matches any AI keyword.

    Two-tier matching:
    - Short acronyms (ai, llm, gpt …): word-boundary regex to avoid
      false positives — 'ai' would otherwise match 'rain' or 'email'.
    - Longer phrases: plain substring match (case-insensitive).
    """
    return bool(_SHORT_KEYWORDS_RE.search(title) or _LONG_KEYWORDS_RE.search(title))


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


class HackerNewsSource(BaseSource):
    source_name = "hackernews"

    MAX_STORIES: int = 200
    _CONCURRENCY: int = 10

    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=20.0,
            )
        return self._client

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get_json(self, url: str) -> object:
        client = await self._get_client()
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

    async def _fetch_story(
        self,
        sem: asyncio.Semaphore,
        story_id: int,
    ) -> dict | None:
        async with sem:
            try:
                data = await self._get_json(f"{_BASE_URL}/item/{story_id}.json")
                return data if isinstance(data, dict) else None
            except Exception as exc:
                logger.warning("Failed to fetch HN item %d: %s", story_id, exc)
                return None

    def _to_item(self, raw: dict) -> SourceItem:
        story_id = raw["id"]
        url = raw.get("url") or _HN_ITEM_URL.format(id=story_id)
        published_at = datetime.fromtimestamp(raw["time"], tz=UTC)
        return SourceItem(
            id=f"hn:{story_id}",
            source=self.source_name,
            title=raw["title"],
            url=url,
            content=raw.get("text"),
            author=raw.get("by"),
            published_at=published_at,
            metadata={
                "score": raw.get("score", 0),
                "comments": raw.get("descendants", 0),
                "hn_type": raw.get("type"),
                "hn_url": _HN_ITEM_URL.format(id=story_id),
            },
        )

    async def fetch(self, *, since: datetime | None = None) -> Sequence[SourceItem]:
        """Fetch top + best HN stories, filtered to AI-related titles.

        Fetches story IDs from topstories.json and beststories.json, deduplicates,
        caps at MAX_STORIES, then fetches each item concurrently (bounded by a
        semaphore). Stories are filtered client-side by AI keyword matching and
        by the `since` date if provided.
        """
        try:
            top_ids, best_ids = await asyncio.gather(
                self._get_json(f"{_BASE_URL}/topstories.json"),
                self._get_json(f"{_BASE_URL}/beststories.json"),
            )

            # Deduplicate while preserving order (top stories first)
            seen_ids: set[int] = set()
            combined: list[int] = []
            for story_id in list(top_ids) + list(best_ids):  # type: ignore[arg-type]
                if story_id not in seen_ids:
                    seen_ids.add(story_id)
                    combined.append(story_id)
                    if len(combined) >= self.MAX_STORIES:
                        break

            logger.info("HN: fetching %d story IDs", len(combined))

            sem = asyncio.Semaphore(self._CONCURRENCY)
            tasks = [self._fetch_story(sem, sid) for sid in combined]
            raw_stories = await asyncio.gather(*tasks)

            items: list[SourceItem] = []
            for raw in raw_stories:
                if raw is None:
                    continue
                # Drop non-stories, deleted/dead items, and items without a title
                if raw.get("type") != "story":
                    continue
                if raw.get("dead") or raw.get("deleted"):
                    continue
                if not raw.get("title"):
                    continue
                # AI relevance filter
                if not _is_ai_related(raw["title"]):
                    continue
                item = self._to_item(raw)
                # since filter (client-side — HN API has no server-side date filter)
                if since and item.published_at and item.published_at <= since:
                    continue
                items.append(item)

            logger.info(
                "HN: %d AI-relevant stories (from %d candidates, since=%s)",
                len(items),
                len(combined),
                since,
            )
            return items

        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None
