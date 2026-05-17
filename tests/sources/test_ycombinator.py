"""Unit tests for YCombinatorSource.

All HTTP calls are mocked with respx — no real YC or Algolia calls.
"""

from datetime import UTC, datetime

import httpx
import pytest
import respx

from ai_intel.sources.ycombinator import YCombinatorSource, _extract_algolia_opts

_YC_URL = "https://www.ycombinator.com/companies"
_ALGOLIA_URL = "https://TESTAPP-dsn.algolia.net/1/indexes/YCCompany_production/query"

# Minimal HTML containing a valid AlgoliaOpts block
_VALID_HTML = """
<html><head></head><body>
<script>window.AlgoliaOpts = {"app": "TESTAPP", "key": "testkey123"};</script>
</body></html>
"""

# HTML that has no AlgoliaOpts at all
_MISSING_OPTS_HTML = "<html><body>No script here</body></html>"

# HTML where the JSON is malformed
_MALFORMED_HTML = '<html><script>window.AlgoliaOpts = {bad json};</script></html>'


def _make_hit(
    slug: str,
    name: str,
    one_liner: str = "AI startup",
    long_description: str | None = None,
    website: str | None = None,
    launched_at: int | None = 1700000000,
    batch: str = "W24",
) -> dict:
    return {
        "id": hash(slug) & 0xFFFFFF,
        "slug": slug,
        "name": name,
        "one_liner": one_liner,
        "long_description": long_description,
        "website": website or f"https://{slug}.com",
        "launched_at": launched_at,
        "batch": batch,
        "tags": ["Artificial Intelligence"],
        "industries": ["Artificial Intelligence"],
        "status": "Active",
        "team_size": 10,
        "all_locations": "San Francisco, CA, USA",
    }


def _algolia_response(hits: list[dict], nb_pages: int = 1, page: int = 0) -> dict:
    return {
        "hits": hits,
        "page": page,
        "nbPages": nb_pages,
        "nbHits": len(hits),
    }


# ---------------------------------------------------------------------------
# _extract_algolia_opts unit tests
# ---------------------------------------------------------------------------


def test_extract_algolia_opts_valid():
    app_id, api_key = _extract_algolia_opts(_VALID_HTML)
    assert app_id == "TESTAPP"
    assert api_key == "testkey123"


def test_extract_algolia_opts_missing_raises():
    with pytest.raises(RuntimeError, match="Could not find window.AlgoliaOpts"):
        _extract_algolia_opts(_MISSING_OPTS_HTML)


def test_extract_algolia_opts_malformed_json_raises():
    with pytest.raises(RuntimeError, match="Failed to parse AlgoliaOpts JSON"):
        _extract_algolia_opts(_MALFORMED_HTML)


# ---------------------------------------------------------------------------
# fetch() — SourceItem mapping
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_basic_item_mapping():
    hit = _make_hit("acme-ai", "Acme AI", one_liner="Build AI apps faster")
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 1
    item = items[0]
    assert item.id == "yc:acme-ai"
    assert item.source == "ycombinator"
    assert item.title == "Acme AI"
    assert item.url == "https://acme-ai.com"


@respx.mock
async def test_fetch_uses_long_description_over_one_liner():
    hit = _make_hit(
        "verbos-ai",
        "Verbos AI",
        one_liner="Short blurb",
        long_description="Detailed description of the product and mission.",
    )
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items[0].content == "Detailed description of the product and mission."


@respx.mock
async def test_fetch_falls_back_to_one_liner_when_no_long_description():
    hit = _make_hit("quick-ai", "Quick AI", one_liner="Do things fast")
    hit.pop("long_description", None)
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items[0].content == "Do things fast"


@respx.mock
async def test_fetch_uses_yc_url_when_no_website():
    hit = _make_hit("no-site", "No Site AI")
    hit["website"] = None
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items[0].url == "https://www.ycombinator.com/companies/no-site"


@respx.mock
async def test_fetch_published_at_from_launched_at():
    hit = _make_hit("ts-ai", "TS AI", launched_at=1700000000)
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    expected = datetime.fromtimestamp(1700000000, tz=UTC)
    assert items[0].published_at == expected
    assert items[0].published_at.tzinfo is not None


@respx.mock
async def test_fetch_none_published_at_when_no_launched_at():
    hit = _make_hit("no-date", "No Date AI", launched_at=None)
    hit.pop("launched_at", None)
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items[0].published_at is None


@respx.mock
async def test_fetch_metadata_fields():
    hit = _make_hit("meta-ai", "Meta AI", batch="S23")
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    m = items[0].metadata
    assert m["batch"] == "S23"
    assert "Artificial Intelligence" in m["tags"]
    assert m["status"] == "Active"
    assert m["team_size"] == 10
    assert m["yc_url"] == "https://www.ycombinator.com/companies/meta-ai"


@respx.mock
async def test_fetch_source_name():
    hit = _make_hit("src-ai", "Src AI")
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items[0].source == "ycombinator"


# ---------------------------------------------------------------------------
# fetch() — pagination and MAX_COMPANIES cap
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_paginates_multiple_pages():
    hit_a = _make_hit("alpha-ai", "Alpha AI")
    hit_b = _make_hit("beta-ai", "Beta AI")

    call_count = 0

    def algolia_side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        import json as _json

        body = _json.loads(request.content)
        page = body.get("page", 0)
        if page == 0:
            return httpx.Response(
                200, json=_algolia_response([hit_a], nb_pages=2, page=0)
            )
        else:
            return httpx.Response(
                200, json=_algolia_response([hit_b], nb_pages=2, page=1)
            )

    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(side_effect=algolia_side_effect)

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    ids = {item.id for item in items}
    assert "yc:alpha-ai" in ids
    assert "yc:beta-ai" in ids


@respx.mock
async def test_fetch_respects_max_companies():
    hits = [_make_hit(f"co-{i}", f"Company {i}") for i in range(5)]
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response(hits))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        source.MAX_COMPANIES = 3
        items = await source.fetch()

    assert len(items) <= 3


@respx.mock
async def test_fetch_deduplicates_repeated_slugs():
    hit = _make_hit("dup-ai", "Dup AI")
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([hit, hit]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert len(items) == 1


@respx.mock
async def test_fetch_empty_hits_returns_empty_list():
    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(
        return_value=httpx.Response(200, json=_algolia_response([]))
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        items = await source.fetch()

    assert items == []


# ---------------------------------------------------------------------------
# fetch() — since filter sends numericFilters
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_since_sends_numeric_filter():
    """When since is provided, the Algolia payload must include numericFilters."""
    hit = _make_hit("new-ai", "New AI", launched_at=1700000000)

    captured_payload: dict = {}

    def capture_side_effect(request: httpx.Request) -> httpx.Response:
        import json as _json

        nonlocal captured_payload
        captured_payload = _json.loads(request.content)
        return httpx.Response(200, json=_algolia_response([hit]))

    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(side_effect=capture_side_effect)

    since = datetime(2023, 1, 1, tzinfo=UTC)

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        await source.fetch(since=since)

    assert "numericFilters" in captured_payload
    since_ts = int(since.timestamp())
    assert any(f"launched_at>{since_ts}" in f for f in captured_payload["numericFilters"])


@respx.mock
async def test_fetch_without_since_omits_numeric_filter():
    """Without since, Algolia payload must NOT include numericFilters."""
    hit = _make_hit("any-ai", "Any AI")
    captured_payload: dict = {}

    def capture_side_effect(request: httpx.Request) -> httpx.Response:
        import json as _json

        nonlocal captured_payload
        captured_payload = _json.loads(request.content)
        return httpx.Response(200, json=_algolia_response([hit]))

    respx.get(_YC_URL).mock(return_value=httpx.Response(200, text=_VALID_HTML))
    respx.post(_ALGOLIA_URL).mock(side_effect=capture_side_effect)

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        await source.fetch(since=None)

    assert "numericFilters" not in captured_payload


# ---------------------------------------------------------------------------
# fetch() — credential extraction failure
# ---------------------------------------------------------------------------


@respx.mock
async def test_fetch_raises_when_algolia_opts_missing():
    """RuntimeError must propagate if AlgoliaOpts is absent from the YC page."""
    respx.get(_YC_URL).mock(
        return_value=httpx.Response(200, text=_MISSING_OPTS_HTML)
    )

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        with pytest.raises(RuntimeError, match="Could not find window.AlgoliaOpts"):
            await source.fetch()


@respx.mock
async def test_fetch_raises_on_yc_page_http_error():
    """A 5xx error from the YC page must raise after retries."""
    respx.get(_YC_URL).mock(return_value=httpx.Response(503))

    async with httpx.AsyncClient() as client:
        source = YCombinatorSource(http_client=client)
        with pytest.raises(httpx.HTTPStatusError):
            await source.fetch()
