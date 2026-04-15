"""
ORION PATCH P1-A — Qdrant optional / graceful degradation
File: orion/memory/longterm.py
Commit: fix(memory): degrade gracefully when Qdrant is unreachable

When Qdrant is not running, the original code raised immediately on import,
crashing the Verifier for any task that touched long-term memory.
This patch:
  1. Wraps the QdrantClient init in a try/except
  2. Falls back to an in-process dict store (ephemeral, but functional)
  3. Exposes is_available() for the /ready endpoint
  4. Logs a clear, actionable warning with the start command
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


class LongTermMemory:
    """
    Semantic vector store backed by Qdrant.
    Falls back to an ephemeral in-process store when Qdrant is unreachable.
    """

    def __init__(self, url: str = "http://localhost:6333", collection: str = "orion_tasks"):
        self._url        = url
        self._collection = collection
        self._client     = None
        self._available  = False
        # Ephemeral fallback: {key: [payload, ...]}
        self._fallback: dict[str, list[dict]] = {}

        self._connect()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            client = QdrantClient(url=self._url, timeout=3, prefer_grpc=False)
            collections = {c.name for c in client.get_collections().collections}

            if self._collection not in collections:
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )

            self._client    = client
            self._available = True
            log.info("Long-term memory: Qdrant connected at %s (collection: %s)",
                     self._url, self._collection)

        except ImportError:
            log.warning(
                "Long-term memory: qdrant-client not installed. "
                "Run `pip install qdrant-client` for persistent memory."
            )
        except Exception as exc:
            log.warning(
                "Long-term memory: Qdrant unreachable at %s (%s).\n"
                "  → Falling back to ephemeral in-process store.\n"
                "  → For persistence, run:  podman-compose up -d qdrant\n"
                "  → Then restart ORION:   make dev",
                self._url, exc,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when backed by a live Qdrant instance."""
        return self._available

    def store(self, key: str, vector: list[float], payload: dict[str, Any]) -> None:
        """Upsert a vector + payload. Falls back to in-process dict when offline."""
        if self._available and self._client:
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name=self._collection,
                points=[PointStruct(id=hash(key) & 0xFFFFFFFF, vector=vector, payload=payload)],
            )
        else:
            # Ephemeral fallback — survives the current process only
            self._fallback.setdefault(key, []).append(payload)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        score_threshold: float = 0.70,
    ) -> list[dict[str, Any]]:
        """
        Return the top-k most similar payloads.
        Falls back to returning the most recent items from the in-process store.
        """
        if self._available and self._client:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            return [hit.payload for hit in hits]
        else:
            # Naive fallback: return the last top_k stored payloads across all keys
            all_payloads = [p for v in self._fallback.values() for p in v]
            return all_payloads[-top_k:]

    def delete(self, key: str) -> None:
        """Delete a stored entry by key."""
        if self._available and self._client:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[hash(key) & 0xFFFFFFFF]),
            )
        else:
            self._fallback.pop(key, None)

    def __repr__(self) -> str:
        backend = f"Qdrant@{self._url}" if self._available else "EphemeralDict(offline)"
        return f"<LongTermMemory backend={backend} collection={self._collection!r}>"
