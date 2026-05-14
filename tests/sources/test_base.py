from datetime import UTC, datetime

import pytest

from ai_intel.sources.base import BaseSource, SourceItem


def test_source_item_defaults():
    item = SourceItem(id="x:1", source="test", title="Title", url="https://example.com")
    assert item.content is None
    assert item.author is None
    assert item.published_at is None
    assert item.metadata == {}


def test_source_item_full():
    now = datetime(2026, 1, 15, tzinfo=UTC)
    item = SourceItem(
        id="x:1",
        source="test",
        title="Title",
        url="https://example.com",
        content="body",
        author="alice",
        published_at=now,
        metadata={"stars": 42},
    )
    assert item.published_at == now
    assert item.metadata["stars"] == 42


def test_missing_source_name_raises():
    with pytest.raises(TypeError, match="source_name"):

        class BadSource(BaseSource):
            async def fetch(self, *, since=None):
                return []


def test_concrete_source_requires_fetch_implementation():
    class GoodSource(BaseSource):
        source_name = "good"

        async def fetch(self, *, since=None):
            return []

    assert GoodSource().source_name == "good"


def test_source_name_inherited_from_parent():
    class ParentSource(BaseSource):
        source_name = "parent"

        async def fetch(self, *, since=None):
            return []

    # Child inherits source_name from parent — no TypeError
    class ChildSource(ParentSource):
        async def fetch(self, *, since=None):
            return []

    assert ChildSource().source_name == "parent"
