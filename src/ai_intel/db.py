from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from ai_intel.config import get_settings


def _build_engine():
    url = get_settings().database_url
    if not url.startswith("postgresql+asyncpg://"):
        raise ValueError(
            f"DATABASE_URL must use the postgresql+asyncpg:// scheme, got: {url!r}"
        )
    return create_async_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )


engine = _build_engine()

session_factory = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


class Base(DeclarativeBase):
    pass


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        yield session
