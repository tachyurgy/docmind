import asyncio
from functools import lru_cache

import structlog
from openai import AsyncOpenAI

from app.config import settings

logger = structlog.get_logger()


class EmbeddingService:
    """Generates text embeddings via OpenAI's API with batching and retry logic."""

    BATCH_SIZE = 100
    MAX_RETRIES = 3
    BASE_DELAY = 1.0

    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimensions = settings.embedding_dimensions
        self._cache: dict[str, list[float]] = {}

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning one vector per input text.

        Uses caching to avoid re-embedding identical strings and batches
        requests to stay within API limits.
        """
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached_indices.append(i)

        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            embeddings = await self._batch_embed(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                self._cache[texts[idx]] = emb
                results[idx] = emb

        return [r for r in results if r is not None]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        result = await self.embed_texts([query])
        return result[0]

    async def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches, respecting API rate limits."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            embeddings = await self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)

            if i + self.BATCH_SIZE < len(texts):
                await asyncio.sleep(0.1)

        return all_embeddings

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call the OpenAI embeddings endpoint with exponential backoff."""
        last_error: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                    dimensions=self._dimensions,
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                last_error = e
                delay = self.BASE_DELAY * (2**attempt)
                logger.warning(
                    "embedding_retry",
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(e),
                )
                await asyncio.sleep(delay)

        raise RuntimeError(f"Embedding failed after {self.MAX_RETRIES} attempts: {last_error}")


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
