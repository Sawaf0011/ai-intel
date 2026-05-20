"""Find AlgoliaOpts in YC page HTML and test the Algolia search endpoint."""
import asyncio
import json
import re

import httpx


async def main() -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ai-intel-scraper/0.1)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=30) as client:
        r = await client.get(
            "https://www.ycombinator.com/companies?industry=Artificial+Intelligence"
        )
        body = r.text

        # Find AlgoliaOpts
        algolia_opts = re.findall(r"AlgoliaOpts[^;]+", body)
        print("AlgoliaOpts lines:", algolia_opts[:5])

        # Find window. assignments
        window_assigns = re.findall(r'window\.[A-Za-z]+\s*=\s*\{[^}]{0,300}\}', body)
        print("window.* assignments:", window_assigns[:5])

        # Find all JSON-looking blocks in script tags
        script_contents = re.findall(r"<script[^>]*>(.*?)</script>", body, re.DOTALL)
        for i, sc in enumerate(script_contents):
            if "algolia" in sc.lower() or "AlgoliaOpts" in sc or "app_id" in sc.lower():
                print(f"Script {i} has algolia refs:", sc[:500])

        # Check the AlgoliaOpts page (different URL)
        r2 = await client.get("https://www.ycombinator.com/companies")
        body2 = r2.text
        algolia2 = re.findall(r"AlgoliaOpts[^;]+", body2)
        print("AlgoliaOpts on /companies:", algolia2[:5])

        # Try fetching the ycdc_new JS which likely sets up Algolia
        r3 = await client.get("https://bookface-static.ycombinator.com/vite/assets/ycdc-styles-D9RPvYit.js")
        body3 = r3.text
        print("ycdc-styles size:", len(body3))
        algolia3 = re.findall(r'(?:AlgoliaOpts|appId|app_id|apiKey)[^;,\n]{0,100}', body3)
        print("Algolia in ycdc-styles:", algolia3[:5])


asyncio.run(main())
