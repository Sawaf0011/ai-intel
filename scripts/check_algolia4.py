"""Test YC Algolia with corrected filter syntax and inspect field structure."""
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
    "User-Agent": "ai-intel-scraper/0.1",
}


async def search(query: str = "", filters: str = "", page: int = 0) -> dict:
    payload: dict = {
        "query": query,
        "hitsPerPage": 5,
        "page": page,
        "attributesToRetrieve": ["*"],
    }
    if filters:
        payload["filters"] = filters
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        url = f"https://{APP_ID}-dsn.algolia.net/1/indexes/{INDEX_NAME}/query"
        r = await client.post(url, json=payload)
        return r.status_code, r.json()


async def main() -> None:
    # Try tagFilters for the public tag (from decoded key we see tagFilters=["ycdc_public"])
    status, data = await search(query="artificial intelligence")
    print("Query 'artificial intelligence' - Status:", status)
    if status == 200:
        print("Total hits:", data.get("nbHits"), "Pages:", data.get("nbPages"))
        hits = data.get("hits", [])
        if hits:
            print("ALL keys on first hit:", sorted(hits[0].keys()))
            print("First hit:")
            print(json.dumps(hits[0], indent=2)[:3000])
    else:
        print("Error:", data)

    print("\n--- Trying facetFilters ---")
    payload = {
        "query": "",
        "facetFilters": [["industry:Artificial Intelligence"]],
        "hitsPerPage": 5,
        "page": 0,
        "attributesToRetrieve": ["*"],
    }
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        url = f"https://{APP_ID}-dsn.algolia.net/1/indexes/{INDEX_NAME}/query"
        r = await client.post(url, json=payload)
        data2 = r.json()
        print("facetFilters status:", r.status_code)
        if r.status_code == 200:
            print("Hits with facetFilters:", data2.get("nbHits"))
            hits2 = data2.get("hits", [])
            if hits2:
                print("First hit name:", hits2[0].get("name"))
                print("First hit keys:", sorted(hits2[0].keys()))


asyncio.run(main())
