"""Confirm Algolia facetFilters on tags and test pagination for YC scraper design."""
import asyncio
import json

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
    # Try tags:"Artificial Intelligence" facetFilter
    print("=== facetFilters: tags:'Artificial Intelligence' ===")
    status, data = await query({
        "query": "",
        "facetFilters": [["tags:Artificial Intelligence"]],
        "hitsPerPage": 10,
        "page": 0,
        "attributesToRetrieve": ["id", "name", "slug", "one_liner", "batch", "tags", "launched_at", "status", "website", "team_size", "all_locations"],
    })
    print("Status:", status)
    if status == 200:
        print("Total hits:", data.get("nbHits"), "Pages:", data.get("nbPages"))
        hits = data.get("hits", [])
        print(f"Page 0 hits: {len(hits)}")
        for h in hits[:3]:
            print(f"  - {h.get('name')} [{h.get('batch')}] tags={h.get('tags')} launched_at={h.get('launched_at')}")
    else:
        print("Error:", data)

    # Test page 2 to confirm pagination works
    print("\n=== Page 1 ===")
    status2, data2 = await query({
        "query": "",
        "facetFilters": [["tags:Artificial Intelligence"]],
        "hitsPerPage": 100,
        "page": 0,
        "attributesToRetrieve": ["id", "name", "slug", "one_liner", "batch", "tags", "launched_at", "status", "website", "team_size", "all_locations"],
    })
    print("Status:", status2)
    if status2 == 200:
        hits2 = data2.get("hits", [])
        print(f"hitsPerPage=100 → got {len(hits2)} hits, total {data2.get('nbHits')}")
        # Show launched_at range
        launch_times = [h.get("launched_at") for h in hits2 if h.get("launched_at")]
        if launch_times:
            print("launched_at range:", min(launch_times), "to", max(launch_times))
            from datetime import datetime, UTC
            print("  =", datetime.fromtimestamp(min(launch_times), tz=UTC).isoformat(),
                  "to", datetime.fromtimestamp(max(launch_times), tz=UTC).isoformat())
        # Show sample
        for h in hits2[:3]:
            print(f"  {h.get('name')} | {h.get('batch')} | {h.get('one_liner', '')[:60]}")
    else:
        print("Error:", data2)

asyncio.run(main())
