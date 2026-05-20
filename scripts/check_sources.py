import asyncio
from ai_intel.db import session_factory
from sqlalchemy import text


async def query():
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT source, COUNT(*) AS total, COUNT(embedding) AS embedded "
                "FROM items GROUP BY source ORDER BY source"
            )
        )
        rows = result.fetchall()
        print(f"{'Source':<15} {'Total':>8} {'Embedded':>10}")
        print("-" * 35)
        for row in rows:
            print(f"{row[0]:<15} {row[1]:>8} {row[2]:>10}")


asyncio.run(query())
