import argparse
import asyncio
import logging
from datetime import UTC, datetime

from ai_intel.config import get_settings
from ai_intel.db import session_factory
from ai_intel.embeddings.pipeline import run_embedding_pipeline
from ai_intel.sources.github import GitHubSource
from ai_intel.sources.hackernews import HackerNewsSource
from ai_intel.sources.runner import run_source
from ai_intel.sources.ycombinator import YCombinatorSource


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD")


async def _scrape(source_name: str, since: datetime | None) -> None:
    settings = get_settings()
    if source_name == "github":
        if not settings.github_token:
            raise SystemExit("GITHUB_TOKEN is required for the github source")
        source = GitHubSource(settings.github_token)
    elif source_name == "hackernews":
        source = HackerNewsSource()
    elif source_name == "ycombinator":
        source = YCombinatorSource()
    else:
        raise SystemExit(f"Unknown source: {source_name!r}")

    async with session_factory() as session:
        count = await run_source(source, session, since=since)

    print(f"Done — upserted {count} items from {source_name}")


async def _embed(force: bool, limit: int | None) -> None:
    result = await run_embedding_pipeline(force=force, limit=limit)
    print(f"Done — embedded {result.embedded} items ({result.total_seen} seen)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="ai-intel",
        description="AI Startup Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # --- scrape subcommand ---
    scrape_parser = subparsers.add_parser(
        "scrape",
        help="Scrape AI startup data into the database",
    )
    scrape_parser.add_argument(
        "--source",
        required=True,
        choices=["github", "hackernews", "ycombinator"],
        help="Data source to scrape",
    )
    scrape_parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        type=_parse_date,
        default=None,
        help="Only fetch items updated after this date (default: resume from last seen)",
    )

    # --- embed subcommand ---
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate vector embeddings for stored items",
    )
    embed_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all items, not just those missing an embedding",
    )
    embed_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of items to embed (useful for testing)",
    )

    args = parser.parse_args()

    if args.command == "scrape":
        asyncio.run(_scrape(args.source, args.since))
    elif args.command == "embed":
        asyncio.run(_embed(force=args.force, limit=args.limit))


if __name__ == "__main__":
    main()
