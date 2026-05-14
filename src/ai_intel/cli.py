import argparse
import asyncio
import logging
from datetime import UTC, datetime

from ai_intel.config import get_settings
from ai_intel.db import session_factory
from ai_intel.sources.github import GitHubSource
from ai_intel.sources.runner import run_source


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
    else:
        raise SystemExit(f"Unknown source: {source_name!r}")

    async with session_factory() as session:
        count = await run_source(source, session, since=since)

    print(f"Done — upserted {count} items from {source_name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="ai-intel-scrape",
        description="Scrape AI startup data into the database",
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=["github"],
        help="Data source to scrape",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        type=_parse_date,
        default=None,
        help="Only fetch items updated after this date (default: resume from last seen)",
    )
    args = parser.parse_args()
    asyncio.run(_scrape(args.source, args.since))


if __name__ == "__main__":
    main()
