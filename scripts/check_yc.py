"""One-off script to investigate YC companies page data path."""
import asyncio
import re

import httpx


async def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-intel-scraper/0.1)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30) as client:
        r = await client.get(
            "https://www.ycombinator.com/companies?industry=Artificial+Intelligence"
        )
        body = r.text
        print("Status:", r.status_code)
        print("Content-Length:", len(body))

        # Algolia references
        algolia_refs = re.findall(r"algolia[\w.\-/]*", body, re.IGNORECASE)
        print("Algolia refs:", algolia_refs[:10])

        # Script src URLs
        scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', body)
        print("Script src URLs:", scripts[:10])

        # Any JSON-looking blocks containing 'companies' or 'batch'
        json_blocks = re.findall(r'\{[^{}]{0,200}(?:batch|company|companies)[^{}]{0,200}\}', body, re.I)
        print("JSON-ish blocks:", json_blocks[:3])

        print("--- Body excerpt (10000-13000) ---")
        print(body[10000:13000])
        print("--- Body end ---")
        print(body[-2000:])


asyncio.run(main())
