"""One-off script to find Algolia credentials in YC JS bundles."""
import asyncio
import re

import httpx


async def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-intel-scraper/0.1)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30) as client:
        r = await client.get(
            "https://bookface-static.ycombinator.com/vite/assets/algoliaClient-BRMfylyj.js"
        )
        body = r.text
        print("Status:", r.status_code, "Size:", len(body))

        # Look for appId, apiKey, indexName patterns
        configs = re.findall(
            r'(?:appId|apiKey|applicationId|searchApiKey|indexName)\s*[=:]\s*["\']([^"\']+)["\']',
            body,
        )
        print("Config values:", configs[:20])

        # Algolia app IDs are uppercase alphanumeric, ~10 chars
        app_ids = re.findall(r'"([A-Z0-9]{10,12})"', body)
        print("Potential app IDs:", list(set(app_ids))[:10])

        # Search API keys are 32-char hex
        api_keys = re.findall(r'"([a-f0-9]{32})"', body)
        print("Potential API keys:", list(set(api_keys))[:5])

        print("--- First 4000 chars ---")
        print(body[:4000])


asyncio.run(main())
