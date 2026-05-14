from datetime import UTC, datetime

import httpx
import respx

from ai_intel.sources.github import GitHubSource

_REPO = {
    "id": 99999,
    "full_name": "owner/ai-repo",
    "html_url": "https://github.com/owner/ai-repo",
    "description": "An AI repository",
    "pushed_at": "2026-01-15T10:00:00Z",
    "owner": {"login": "owner"},
    "stargazers_count": 500,
    "forks_count": 42,
    "language": "Python",
    "topics": ["artificial-intelligence", "llm"],
    "watchers_count": 500,
}


@respx.mock
async def test_fetch_single_page():
    respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [_REPO], "total_count": 1})
    )
    async with httpx.AsyncClient() as client:
        source = GitHubSource("test-token", http_client=client)
        items = await source.fetch()

    assert len(items) == 1
    item = items[0]
    assert item.id == "github:99999"
    assert item.source == "github"
    assert item.title == "owner/ai-repo"
    assert item.url == "https://github.com/owner/ai-repo"
    assert item.content == "An AI repository"
    assert item.author == "owner"
    assert item.published_at == datetime(2026, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert item.metadata["stars"] == 500
    assert item.metadata["language"] == "Python"
    assert "artificial-intelligence" in item.metadata["topics"]


@respx.mock
async def test_fetch_stops_at_partial_page():
    # 50 items < _PER_PAGE (100) → no further pages requested
    repos = [dict(_REPO, id=i) for i in range(50)]
    respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": repos, "total_count": 50})
    )
    async with httpx.AsyncClient() as client:
        source = GitHubSource("test-token", http_client=client)
        items = await source.fetch()

    assert len(items) == 50


@respx.mock
async def test_fetch_empty_response():
    respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    async with httpx.AsyncClient() as client:
        source = GitHubSource("test-token", http_client=client)
        items = await source.fetch()

    assert items == []


@respx.mock
async def test_fetch_since_adds_pushed_filter():
    route = respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [], "total_count": 0})
    )
    since = datetime(2026, 1, 1, tzinfo=UTC)
    async with httpx.AsyncClient() as client:
        source = GitHubSource("test-token", http_client=client)
        await source.fetch(since=since)

    called_url = str(route.calls[0].request.url)
    assert "pushed%3A%3E2026-01-01" in called_url or "pushed:>2026-01-01" in called_url


@respx.mock
async def test_fetch_none_description_becomes_none_content():
    repo = dict(_REPO, description=None)
    respx.get("https://api.github.com/search/repositories").mock(
        return_value=httpx.Response(200, json={"items": [repo], "total_count": 1})
    )
    async with httpx.AsyncClient() as client:
        source = GitHubSource("test-token", http_client=client)
        items = await source.fetch()

    assert items[0].content is None
