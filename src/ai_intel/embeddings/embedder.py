"""OpenAI embeddings wrapper with batching, retry, and dimension validation."""

import logging

import openai
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from ai_intel.config import get_settings
from ai_intel.models.item import Item

logger = logging.getLogger(__name__)

BATCH_SIZE: int = 100
EMBEDDING_DIM: int = 1536


def _is_retryable(exc: BaseException) -> bool:
    """Retry on rate-limit, server errors, and connection errors; not on auth failures."""
    if isinstance(exc, openai.AuthenticationError):
        return False
    return isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIConnectionError,
            openai.APITimeoutError,
        ),
    )


def build_embedding_text(item: Item) -> str:
    """Compose a clean, compact text to embed for an item.

    Format:
        {title}

        {content or ""}

        Source: {source}
        {compact metadata: "Language: Python. Topics: llm, agents. Stars: 12000."}

    Metadata fields pulled for GitHub items: language, topics (up to 5),
    stars, forks, open_issues. Other sources get whatever keys they have.
    Raw JSON is never embedded.
    """
    parts = [item.title, ""]
    if item.content:
        parts.append(item.content)
    parts.extend(["", f"Source: {item.source}"])

    meta = item.metadata_
    meta_parts: list[str] = []
    if lang := meta.get("language"):
        meta_parts.append(f"Language: {lang}")
    if topics := meta.get("topics"):
        if isinstance(topics, list) and topics:
            meta_parts.append(f"Topics: {', '.join(str(t) for t in topics[:5])}")
    if stars := meta.get("stars"):
        meta_parts.append(f"Stars: {stars}")
    if forks := meta.get("forks"):
        meta_parts.append(f"Forks: {forks}")
    if open_issues := meta.get("open_issues"):
        meta_parts.append(f"Open issues: {open_issues}")
    if meta_parts:
        parts.append(". ".join(meta_parts) + ".")

    return "\n".join(parts).strip()


class Embedder:
    BATCH_SIZE: int = BATCH_SIZE
    EMBEDDING_DIM: int = EMBEDDING_DIM

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embed_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches; preserves input order across batches.

        Returns an empty list for empty input without making any API call.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_batches = (len(texts) + self.BATCH_SIZE - 1) // self.BATCH_SIZE

        for batch_num, start in enumerate(range(0, len(texts), self.BATCH_SIZE), 1):
            batch = texts[start : start + self.BATCH_SIZE]
            logger.info(
                "Embedding batch %d/%d (%d texts)", batch_num, total_batches, len(batch)
            )
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings API for a single batch; order-preserving."""
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        # API may return data out of order — sort by index to guarantee alignment.
        sorted_data = sorted(response.data, key=lambda d: d.index)
        embeddings = [d.embedding for d in sorted_data]
        for vec in embeddings:
            if len(vec) != self.EMBEDDING_DIM:
                raise ValueError(
                    f"Expected {self.EMBEDDING_DIM}-dim vector from model "
                    f"'{self._model}', got {len(vec)}. "
                    "Check OPENAI_EMBED_MODEL in settings."
                )
        return embeddings
