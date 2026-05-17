"""Y Combinator companies scraper via the Algolia search index.

Data path investigation findings (2026-05-17):
  - https://www.ycombinator.com/companies is a client-side React/Inertia app.
    A plain HTTP GET returns an HTML shell with no company data.
  - Algolia credentials ARE embedded in the page HTML as:
        window.AlgoliaOpts = {"app": "<APP_ID>", "key": "<SEARCH_KEY>"};
    These are public, restricted read-only keys (restrictIndices, tagFilters
    embedded in the key itself). Extracted with a regex each run so they
    stay current if YC rotates them.
  - Index: YCCompany_production
  - Filter: facetFilters=[["tags:Artificial Intelligence"]] -> ~928 AI companies
  - since support: numericFilters=["launched_at > <unix_ts>"]
  - selectolax is NOT required — credentials are in a <script> block, parsed
    with json.loads(); company data comes back as clean JSON from Algolia.

No authentication required beyond the public key embedded in the page.
"""

import asyncio
import json
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

_YC_COMPANIES_URL = "https://www.ycombinator.com/companies"
_ALGOLIA_INDEX = "YCCompany_production"
_USER_AGENT = "ai-intel-scraper/0.1 (+https://github.com/placeholder/ai-intel)"

# Pattern that matches the AlgoliaOpts assignment in the page's inline script.
# The value is a JSON object terminated by a semicolon.
_ALGOLIA_OPTS_RE = re.compile(r"window\.AlgoliaOpts\s*=\s*(\{[^;]+\})")

# Polite delay between Algolia page requests (seconds).
_PAGE_DELAY = 1.2


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))


def _extract_algolia_opts(html: str) -> tuple[str, str]:
    """Parse app ID and search key from YC page HTML.

    Raises RuntimeError with an actionable message if the pattern is not found,
    so a future page change produces a clear failure rather than a silent crash.
    """
    match = _ALGOLIA_OPTS_RE.search(html)
    if not match:
        raise RuntimeError(
            "Could not find window.AlgoliaOpts in the YC companies page. "
            "The page structure may have changed. "
            "Check https://www.ycombinator.com/companies and update _ALGOLIA_OPTS_RE."
        )
    try:
        opts = json.loads(match.group(1))
        return opts["app"], opts["key"]
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"Failed to parse AlgoliaOpts JSON from YC page: {exc}. "
            f"Raw match: {match.group(1)[:200]!r}"
        ) from exc


class YCombinatorSource(BaseSource):
    """Scrape AI-tagged companies from the YC public directory via Algolia.

    Credentials are fetched fresh from the YC page on each run, so they stay
    current if YC rotates the public search key.
    """

    source_name = "ycombinator"

    MAX_COMPANIES: int = 200
    _HITS_PER_PAGE: int = 100

    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": _USER_AGENT},
                timeout=30.0,
            )
        return self._client

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _get(self, url: str) -> httpx.Response:
        client = await self._get_client()
        r = await client.get(url)
        r.raise_for_status()
        return r

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _algolia_query(
        self,
        algolia_url: str,
        headers: dict,
        payload: dict,
    ) -> dict:
        client = await self._get_client()
        r = await client.post(algolia_url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()

    def _to_item(self, hit: dict) -> SourceItem:
        slug = hit.get("slug") or str(hit["id"])
        yc_url = f"https://www.ycombinator.com/companies/{slug}"
        launched_at = hit.get("launched_at")
        published_at = (
            datetime.fromtimestamp(launched_at, tz=UTC) if launched_at else None
        )
        # Use long_description if present; fall back to one_liner.
        content = hit.get("long_description") or hit.get("one_liner")
        return SourceItem(
            id=f"yc:{slug}",
            source=self.source_name,
            title=hit["name"],
            url=hit.get("website") or yc_url,
            content=content,
            author=None,
            published_at=published_at,
            metadata={
                "batch": hit.get("batch"),
                "industry": hit.get("industries") or hit.get("industry"),
                "tags": hit.get("tags", []),
                "team_size": hit.get("team_size"),
                "status": hit.get("status"),
                "website": hit.get("website"),
                "location": hit.get("all_locations"),
                "one_liner": hit.get("one_liner"),
                "yc_url": yc_url,
            },
        )

    async def fetch(self, *, since: datetime | None = None) -> Sequence[SourceItem]:
        """Fetch AI-tagged YC companies from the Algolia index.

        Extracts Algolia credentials from the live YC page on each call so they
        stay current. Paginates through results in batches of 100 with a polite
        delay between pages. Stops once MAX_COMPANIES items have been collected.

        since: if provided, filters via Algolia numericFilters on launched_at
               (Unix timestamp). Note that launched_at is the YC batch launch
               date — this is the best available proxy but is not a last-modified
               timestamp. Running with since=None returns all AI companies.
        """
        try:
            # --- Step 1: extract credentials from the live page ---
            logger.info("YC: fetching Algolia credentials from %s", _YC_COMPANIES_URL)
            page_resp = await self._get(_YC_COMPANIES_URL)
            app_id, api_key = _extract_algolia_opts(page_resp.text)
            logger.info("YC: Algolia app_id=%s", app_id)

            algolia_url = (
                f"https://{app_id}-dsn.algolia.net/1/indexes/{_ALGOLIA_INDEX}/query"
            )
            algolia_headers = {
                "X-Algolia-Application-Id": app_id,
                "X-Algolia-API-Key": api_key,
                "Content-Type": "application/json",
            }

            # --- Step 2: paginate through Algolia results ---
            numeric_filters: list[str] = []
            if since:
                since_ts = int(since.timestamp())
                numeric_filters.append(f"launched_at>{since_ts}")

            items: list[SourceItem] = []
            seen_ids: set[str] = set()
            page = 0

            while len(items) < self.MAX_COMPANIES:
                payload: dict = {
                    "query": "",
                    "facetFilters": [["tags:Artificial Intelligence"]],
                    "hitsPerPage": self._HITS_PER_PAGE,
                    "page": page,
                    "attributesToRetrieve": [
                        "id",
                        "name",
                        "slug",
                        "one_liner",
                        "long_description",
                        "batch",
                        "tags",
                        "industries",
                        "industry",
                        "launched_at",
                        "status",
                        "website",
                        "team_size",
                        "all_locations",
                    ],
                }
                if numeric_filters:
                    payload["numericFilters"] = numeric_filters

                data = await self._algolia_query(algolia_url, algolia_headers, payload)

                hits = data.get("hits", [])
                nb_pages = data.get("nbPages", 1)

                logger.info(
                    "YC: page %d/%d — %d hits (total so far: %d)",
                    page + 1,
                    nb_pages,
                    len(hits),
                    len(items),
                )

                for hit in hits:
                    if len(items) >= self.MAX_COMPANIES:
                        break
                    item_id = f"yc:{hit.get('slug') or hit['id']}"
                    if item_id in seen_ids:
                        continue
                    seen_ids.add(item_id)
                    items.append(self._to_item(hit))

                if page + 1 >= nb_pages or not hits:
                    break

                page += 1
                # Polite delay between page requests
                await asyncio.sleep(_PAGE_DELAY)

            logger.info(
                "YC: collected %d AI companies (since=%s)", len(items), since
            )
            return items

        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None
