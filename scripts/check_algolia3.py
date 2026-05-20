"""Test the YC Algolia search endpoint with the extracted credentials."""
import asyncio
import base64
import json

import httpx

APP_ID = "45BWZJ1SGC"
API_KEY = "NzllNTY5MzJiZGM2OTY2ZTQwMDEzOTNhYWZiZGRjODlhYzVkNjBmOGRjNzJiMWM4ZTU0ZDlhYTZjOTJiMjlhMWFuYWx5dGljc1RhZ3M9eWNkYyZyZXN0cmljdEluZGljZXM9WUNDb21wYW55X3Byb2R1Y3Rpb24lMkNZQ0NvbXBhbnlfQnlfTGF1bmNoX0RhdGVfcHJvZHVjdGlvbiZ0YWdGaWx0ZXJzPSU1QiUyMnljZGNfcHVibGljJTIyJTVE"
INDEX_NAME = "YCCompany_production"


async def main() -> None:
    # Decode the base64 key to understand its restrictions
    try:
        decoded = base64.b64decode(API_KEY + "==").decode("utf-8", errors="replace")
        print("Decoded key prefix:", decoded[:200])
    except Exception as e:
        print("Key decode error:", e)

    headers = {
        "X-Algolia-Application-Id": APP_ID,
        "X-Algolia-API-Key": API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "ai-intel-scraper/0.1",
    }

    # Test search for AI companies
    payload = {
        "query": "",
        "filters": "industry:Artificial Intelligence",
        "hitsPerPage": 5,
        "page": 0,
        "attributesToRetrieve": [
            "id", "name", "slug", "one_liner", "long_description",
            "batch", "status", "industry", "tags", "team_size",
            "website", "yc_url", "location", "launched_at",
        ],
    }

    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        url = f"https://{APP_ID}-dsn.algolia.net/1/indexes/{INDEX_NAME}/query"
        r = await client.post(url, json=payload)
        print("Status:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Total hits:", data.get("nbHits"))
            print("Total pages:", data.get("nbPages"))
            print("Hits per page:", data.get("hitsPerPage"))
            hits = data.get("hits", [])
            print(f"Returned {len(hits)} hits")
            if hits:
                print("First hit keys:", list(hits[0].keys()))
                print("First hit:", json.dumps(hits[0], indent=2)[:2000])
        else:
            print("Error body:", r.text[:500])


asyncio.run(main())
