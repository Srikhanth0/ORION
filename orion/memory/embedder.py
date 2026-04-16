"""Embedder — singleton semantic embedding pipeline.

Uses sentence-transformers all-MiniLM-L6-v2 (384-dim) with LRU
caching for repeated queries.

Module Contract
---------------
- **Inputs**: text string(s).
- **Outputs**: float vectors (384-dim).

Depends On
----------
- ``sentence_transformers`` (SentenceTransformer)
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_CACHE_SIZE = 1000


class Embedder:
    """Singleton semantic embedding pipeline.

    Loads all-MiniLM-L6-v2 once at first use, caches the last
    1000 encoded strings via LRU to avoid redundant inference.

    Args:
        model_name: Sentence-transformer model name.
        cache_size: Max LRU cache entries.
    """

    _instance: Embedder | None = None

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_size: int = _CACHE_SIZE,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size

    @classmethod
    def get_instance(cls, model_name: str = _DEFAULT_MODEL) -> Embedder:
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import (
                SentenceTransformer,
            )

            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "embedder_loaded",
                model=self._model_name,
            )
        except ImportError:
            logger.warning("sentence_transformers_unavailable")
            self._model = None

    def encode(self, text: str) -> list[float]:
        """Encode a single text string.

        Args:
            text: Input text.

        Returns:
            384-dim float vector.
        """
        cache_key = self._hash(text)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        self._load_model()

        if self._model is None:
            return [0.0] * 384

        embedding = self._model.encode([text], show_progress_bar=False)[0].tolist()

        self._cache[cache_key] = embedding
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return embedding  # type: ignore[no-any-return]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts at once.

        Uses cache for already-seen inputs, batches the rest.

        Args:
            texts: List of input strings.

        Returns:
            List of 384-dim float vectors.
        """
        results: list[list[float] | None] = [None] * len(texts)
        to_encode: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                self._cache.move_to_end(key)
                results[i] = self._cache[key]
            else:
                to_encode.append((i, text))

        if to_encode:
            self._load_model()

            if self._model is None:
                for idx, _ in to_encode:
                    results[idx] = [0.0] * 384
            else:
                batch_texts = [t for _, t in to_encode]
                embeddings = self._model.encode(batch_texts, show_progress_bar=False)

                for j, (idx, text) in enumerate(to_encode):
                    vec = embeddings[j].tolist()
                    key = self._hash(text)
                    self._cache[key] = vec
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)
                    results[idx] = vec

        return [r if r is not None else [0.0] * 384 for r in results]

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    @staticmethod
    def _hash(text: str) -> str:
        """Create a short hash key for cache lookup."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
