from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar


@dataclass
class SourceItem:
    id: str
    source: str
    title: str
    url: str
    content: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    metadata: dict = field(default_factory=dict)


class BaseSource(ABC):
    source_name: ClassVar[str]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "source_name"):
            raise TypeError(f"{cls.__name__} must define class attribute 'source_name'")

    @abstractmethod
    async def fetch(self, *, since: datetime | None = None) -> Sequence[SourceItem]: ...
