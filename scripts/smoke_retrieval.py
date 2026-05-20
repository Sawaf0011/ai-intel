"""Part A retrieval smoke test — run against the live DB."""

import asyncio
import json


async def main() -> None:
    from ai_intel.retrieval.tools import (
        compare_sources,
        get_item_details,
        get_trending,
        search_knowledge_base,
    )

    sep = "-" * 60

    # 1. Semantic search — all sources
    print(sep)
    print("1. search_knowledge_base('AI agent frameworks', limit=5)")
    results = await search_knowledge_base("AI agent frameworks", limit=5)
    for r in results:
        print(f"  [{r['source']:12}] sim={r['similarity']:.3f}  {r['title'][:60]}")
    print()

    # 2. Source-filtered search
    print(sep)
    print("2. search_knowledge_base('open source LLM', source='github', limit=5)")
    results = await search_knowledge_base("open source LLM", source="github", limit=5)
    all_github = all(r["source"] == "github" for r in results)
    print(f"  All source==github: {all_github}")
    for r in results:
        print(f"  sim={r['similarity']:.3f}  {r['title'][:60]}")
    print()

    # 3. Trending GitHub by stars
    print(sep)
    print("3. get_trending('github', timeframe_days=30, limit=5)")
    items = await get_trending("github", timeframe_days=30, limit=5)
    for it in items:
        stars = it["metadata"].get("stars", "?")
        print(f"  stars={stars:>7}  {it['title'][:55]}")
    print()

    # 4. Trending HN by score
    print(sep)
    print("4. get_trending('hackernews', timeframe_days=90, limit=5)")
    items = await get_trending("hackernews", timeframe_days=90, limit=5)
    for it in items:
        score = it["metadata"].get("score", "?")
        print(f"  score={score:>5}  {it['title'][:55]}")
    print()

    # 5. get_item_details
    print(sep)
    results = await search_knowledge_base("LLM inference", limit=1)
    if results:
        item_id = results[0]["id"]
        print(f"5. get_item_details('{item_id}')")
        detail = await get_item_details(item_id)
        print(f"  title:   {detail['title']}")
        print(f"  source:  {detail['source']}")
        content_preview = (detail["content"] or "")[:120]
        print(f"  content: {content_preview!r}")
        print(f"  metadata keys: {list(detail['metadata'].keys())}")
    print()

    # 6. compare_sources
    print(sep)
    print("6. compare_sources('AI safety', limit_per_source=3)")
    compared = await compare_sources("AI safety", limit_per_source=3)
    for src, items in compared.items():
        print(f"  {src} ({len(items)} results):")
        for it in items:
            print(f"    sim={it['similarity']:.3f}  {it['title'][:50]}")
    print()


asyncio.run(main())
