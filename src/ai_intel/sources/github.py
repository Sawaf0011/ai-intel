import asyncio
import logging
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

_SEARCH_URL = "https://api.github.com/search/repositories"
_QUERY = (
    "topic:llm fork:false archived:false stars:>20",
    "topic:generative-ai fork:false archived:false stars:>20",
    "topic:machine-learning fork:false archived:false stars:>20",
)
_PER_PAGE = 100
_MAX_PAGES = 5


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (403, 429, 500, 502, 503, 504)
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


class GitHubSource(BaseSource):
    source_name = "github"

    def __init__(
        self,
        token: str,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
                timeout=30.0,
            )
        return self._client

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _fetch_page(self, page: int, query: str) -> dict:
        client = await self._get_client()

        response = await client.get(
            _SEARCH_URL,
            params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": _PER_PAGE,
                "page": page,
            },
        )

        if response.status_code in (403, 429):
            remaining = int(response.headers.get("X-RateLimit-Remaining", 1))

            if remaining == 0:
                reset_ts = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_s = max(
                    reset_ts - int(datetime.now(UTC).timestamp()),
                    1,
                )

                logger.warning(
                    "GitHub rate limit hit; waiting %ds for reset",
                    wait_s,
                )

                await asyncio.sleep(wait_s)

        response.raise_for_status()

        return response.json()

    def _to_item(self, repo: dict) -> SourceItem:
        pushed = repo.get("pushed_at")
        published_at: datetime | None = None
        if pushed:
            published_at = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
        return SourceItem(
            id=f"github:{repo['id']}",
            source=self.source_name,
            title=repo["full_name"],
            url=repo["html_url"],
            content=repo.get("description"),
            author=repo.get("owner", {}).get("login"),
            published_at=published_at,
            metadata={
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language"),
                "topics": repo.get("topics", []),
                "watchers": repo.get("watchers_count", 0),
            },
        )

    async def fetch(self, *, since: datetime | None = None) -> Sequence[SourceItem]:
        pushed_after = since.strftime("%Y-%m-%d") if since else None

        items: list[SourceItem] = []

        try:
            for base_query in _QUERY:
                query = base_query

                if pushed_after:
                    query += f" pushed:>{pushed_after}"

                for page in range(1, _MAX_PAGES + 1):
                    data = await self._fetch_page(page, query)

                    repos = data.get("items", [])

                    items.extend(self._to_item(r) for r in repos)

                    logger.info(
                        "GitHub query=%s page=%d repos=%d total=%d",
                        base_query,
                        page,
                        len(repos),
                        len(items),
                    )

                    if len(repos) < _PER_PAGE:
                        break

            # Deduplicate repositories
            unique_items = {item.id: item for item in items}

            return list(unique_items.values())

        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None
