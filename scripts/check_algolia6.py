"""Confirm hitsPerPage=100 and numericFilters for since filtering."""
import asyncio
from datetime import UTC, datetime

import httpx

APP_ID = "45BWZJ1SGC"
API_KEY = "NzllNTY5MzJiZGM2OTY2ZTQwMDEzOTNhYWZiZGRjODlhYzVkNjBmOGRjNzJiMWM4ZTU0ZDlhYTZjOTJiMjlhMWFuYWx5dGljc1RhZ3M9eWNkYyZyZXN0cmljdEluZGljZXM9WUNDb21wYW55X3Byb2R1Y3Rpb24lMkNZQ0NvbXBhbnlfQnlfTGF1bmNoX0RhdGVfcHJvZHVjdGlvbiZ0YWdGaWx0ZXJzPSU1QiUyMnljZGNfcHVibGljJTIyJTVE"
INDEX_NAME = "YCCompany_production"
HEADERS = {
    "X-Algolia-Application-Id": APP_ID,
    "X-Algolia-API-Key": API_KEY,
    "Content-Type": "application/json",
}


async def query(payload: dict) -> tuple[int, dict]:
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        url = f"https://{APP_ID}-dsn.algolia.net/1/indexes/{INDEX_NAME}/query"
        r = await client.post(url, json=payload)
        return r.status_code, r.json()


async def main() -> None:
    # Test hitsPerPage=100
    status, data = await query({
        "query": "",
        "facetFilters": [["tags:Artificial Intelligence"]],
        "hitsPerPage": 100,
        "page": 0,
        "attributesToRetrieve": ["id", "name", "slug", "one_liner", "batch", "launched_at", "status"],
    })
    print(f"hitsPerPage=100: status={status}, nbHits={data.get('nbHits')}, got={len(data.get('hits', []))}")

    hits = data.get("hits", [])
    launch_times = [h.get("launched_at") for h in hits if h.get("launched_at")]
    if launch_times:
        print(f"  launched_at range: {datetime.fromtimestamp(min(launch_times), tz=UTC).date()} to {datetime.fromtimestamp(max(launch_times), tz=UTC).date()}")

    # Test numericFilters for since (e.g., since 2024-01-01)
    since_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp())
    print(f"\nTesting numericFilters for launched_at > {since_ts} (2024-01-01):")
    status2, data2 = await query({
        "query": "",
        "facetFilters": [["tags:Artificial Intelligence"]],
        "numericFilters": [f"launched_at>{since_ts}"],
        "hitsPerPage": 20,
        "page": 0,
        "attributesToRetrieve": ["id", "name", "batch", "launched_at"],
    })
    print(f"  status={status2}, nbHits={data2.get('nbHits')}")
    hits2 = data2.get("hits", [])
    for h in hits2[:3]:
        ts = h.get("launched_at", 0)
        dt = datetime.fromtimestamp(ts, tz=UTC).isoformat() if ts else "none"
        print(f"  {h.get('name')} [{h.get('batch')}] launched={dt}")

asyncio.run(main())
