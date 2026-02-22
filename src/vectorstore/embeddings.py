"""
AinSeba - Embedding Generator
Wraps OpenAI's text-embedding-3-small for generating document and query embeddings.
"""

import logging
import time
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using OpenAI's text-embedding-3-small model.
    
    Features:
    - Batch embedding with automatic chunking (API limit: 2048 texts/request)
    - Retry logic for rate limits
    - Token usage tracking
    - Configurable model and dimensions
    """

    # OpenAI batch limit
    MAX_BATCH_SIZE = 2048

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
    ):
        """
        Args:
            api_key: OpenAI API key.
            model: Embedding model name.
            dimensions: Optional output dimensions (text-embedding-3-small supports 512, 1536).
                       None uses the model's default (1536).
        """
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY in your .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.total_tokens_used = 0

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts per API call (max 2048).
            show_progress: Whether to log progress.

        Returns:
            List of embedding vectors (same order as input texts).
        """
        if not texts:
            return []

        batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        logger.info(
            f"Embedding {len(texts)} texts in {total_batches} batches "
            f"(model: {self.model})"
        )

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            if show_progress:
                logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)")

            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        logger.info(
            f"Embedding complete. Total tokens used: {self.total_tokens_used:,}"
        )

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a single query string.

        Args:
            query: The query text.

        Returns:
            Embedding vector.
        """
        result = self._embed_batch([query])
        return result[0]

    def _embed_batch(
        self,
        texts: list[str],
        max_retries: int = 3,
    ) -> list[list[float]]:
        """
        Embed a single batch with retry logic.
        
        Args:
            texts: Batch of texts to embed.
            max_retries: Number of retries on rate limit errors.

        Returns:
            List of embedding vectors.
        """
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "input": texts,
                }
                if self.dimensions is not None:
                    kwargs["dimensions"] = self.dimensions

                response = self.client.embeddings.create(**kwargs)

                # Track token usage
                self.total_tokens_used += response.usage.total_tokens

                # Extract embeddings in order
                embeddings = [item.embedding for item in response.data]
                return embeddings

            except Exception as e:
                error_str = str(e).lower()

                # Rate limit — wait and retry
                if "rate_limit" in error_str or "429" in error_str:
                    wait_time = 2 ** (attempt + 1)  # Exponential backoff
                    logger.warning(
                        f"Rate limited. Waiting {wait_time}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                # Other errors — raise immediately
                logger.error(f"Embedding error: {e}")
                raise

        raise RuntimeError(
            f"Failed to embed batch after {max_retries} retries (rate limited)"
        )

    def get_embedding_dimension(self) -> int:
        """Get the output dimension of the configured model."""
        if self.dimensions is not None:
            return self.dimensions

        # Default dimensions for known models
        defaults = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return defaults.get(self.model, 1536)
