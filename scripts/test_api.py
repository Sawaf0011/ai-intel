"""Test the /query API endpoint against a running server."""

import asyncio
import httpx


async def main() -> None:
    base = "http://localhost:8001"

    async with httpx.AsyncClient(timeout=90.0) as client:
        # Test /health
        r = await client.get(f"{base}/health")
        print(f"/health -> {r.status_code} {r.json()}")

        # Test /query with valid question
        r = await client.post(
            f"{base}/query",
            json={"question": "What are notable open source LLM projects?"},
        )
        print(f"\nPOST /query -> {r.status_code}")
        body = r.json()
        print(f"  iterations:       {body['iterations']}")
        print(f"  tool_calls count: {len(body['tool_calls'])}")
        print(f"  sources count:    {len(body['sources'])}")
        print(f"  tool names:       {[tc['name'] for tc in body['tool_calls']]}")
        print(f"  answer (first 200): {body['answer'][:200]}")

        # Test 422 — empty question
        r = await client.post(f"{base}/query", json={"question": ""})
        print(f"\nPOST /query (empty question) -> {r.status_code}")

        # Test 422 — missing field
        r = await client.post(f"{base}/query", json={})
        print(f"POST /query (no field)       -> {r.status_code}")


asyncio.run(main())
