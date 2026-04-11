"""LongTermMemory — persistent cross-task memory backed by Qdrant.

Stores successful task results with embeddings for semantic retrieval.
Failed tasks go to a separate failure collection.

Module Contract
---------------
- **Inputs**: TaskResult objects.
- **Outputs**: PastTask retrieval via cosine similarity.

Depends On
----------
- ``qdrant_client`` (QdrantClient)
- ``orion.memory.embedder`` (Embedder)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from orion.memory.embedder import Embedder

logger = structlog.get_logger(__name__)

_DEFAULT_COLLECTION = "orion_tasks"
_FAILURE_COLLECTION = "orion_failures"


@dataclass
class PastTask:
    """A retrieved past task from long-term memory.

    Attributes:
        task_description: Original task description.
        execution_plan: Serialised TaskDAG.
        step_results_summary: Brief outcome per step.
        success: Whether the task succeeded.
        duration_seconds: Task duration.
        tools_used: List of tools used.
        timestamp: ISO 8601 timestamp.
        score: Retrieval similarity score.
        doc_id: Document ID in the store.
    """

    task_description: str
    execution_plan: dict[str, Any] = field(
        default_factory=dict
    )
    step_results_summary: str = ""
    success: bool = True
    duration_seconds: float = 0.0
    tools_used: list[str] = field(default_factory=list)
    timestamp: str = ""
    score: float = 0.0
    doc_id: str = ""


class LongTermMemory:
    """Persistent cross-task memory backed by Qdrant.

    Stores successful tasks in the main collection and
    failures in a separate collection. Provides semantic
    retrieval for few-shot examples.

    Args:
        url: Qdrant server URL.
        collection: Main collection name.
        api_key: Qdrant API key (optional).
        embedder: Embedder instance.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = _DEFAULT_COLLECTION,
        api_key: str | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._url = url
        self._collection = collection
        self._failure_collection = _FAILURE_COLLECTION
        self._api_key = api_key
        self._embedder = embedder or Embedder.get_instance()
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-load the Qdrant client."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self._url,
                api_key=self._api_key,
            )
            self._ensure_collections()
            logger.info(
                "qdrant_connected",
                url=self._url,
                collection=self._collection,
            )
        except Exception as exc:
            logger.warning(
                "qdrant_connection_failed",
                url=self._url,
                error=str(exc),
            )
            self._client = None

        return self._client

    def _ensure_collections(self) -> None:
        """Create collections if they don't exist."""
        from qdrant_client.models import (
            Distance,
            VectorParams,
        )

        for name in (
            self._collection,
            self._failure_collection,
        ):
            try:
                self._client.get_collection(name)
            except Exception:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=384,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "collection_created",
                    collection=name,
                )

    def store(
        self,
        task_description: str,
        execution_plan: dict[str, Any],
        step_results_summary: str,
        success: bool,
        duration_seconds: float = 0.0,
        tools_used: list[str] | None = None,
        agent_versions: dict[str, str] | None = None,
    ) -> str:
        """Store a completed task in long-term memory.

        PASS and SOFT_FAIL-with-recovery go to the main
        collection. HARD_FAIL goes to the failure collection.

        Args:
            task_description: Original instruction.
            execution_plan: Serialised TaskDAG.
            step_results_summary: Outcome summary.
            success: Whether the task succeeded.
            duration_seconds: Total task duration.
            tools_used: Tools used in execution.
            agent_versions: Agent version map.

        Returns:
            Document ID assigned by the store.
        """
        doc_id = str(uuid.uuid4())
        embedding = self._embedder.encode(task_description)

        payload = {
            "task_description": task_description,
            "execution_plan": json.dumps(execution_plan),
            "step_results_summary": step_results_summary,
            "success": success,
            "duration_seconds": duration_seconds,
            "tools_used": tools_used or [],
            "timestamp": datetime.now(
                tz=UTC
            ).isoformat(),
            "agent_versions": agent_versions or {},
        }

        collection = (
            self._collection if success
            else self._failure_collection
        )

        client = self._get_client()
        if client is None:
            logger.warning(
                "longterm_store_skipped",
                reason="no_qdrant_connection",
            )
            return doc_id

        try:
            from qdrant_client.models import PointStruct

            client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload,
                    )
                ],
            )

            logger.info(
                "longterm_stored",
                doc_id=doc_id,
                collection=collection,
                success=success,
            )
        except Exception as exc:
            logger.error(
                "longterm_store_failed",
                error=str(exc),
            )

        return doc_id

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        score_threshold: float = 0.5,
    ) -> list[PastTask]:
        """Retrieve similar past tasks.

        Args:
            query: Natural language query.
            top_k: Maximum results.
            score_threshold: Minimum similarity.

        Returns:
            List of PastTask sorted by score.
        """
        embedding = self._embedder.encode(query)

        client = self._get_client()
        if client is None:
            return []

        try:
            results = client.search(
                collection_name=self._collection,
                query_vector=embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )

            past_tasks: list[PastTask] = []
            for hit in results:
                payload = hit.payload or {}
                plan_str = payload.get(
                    "execution_plan", "{}"
                )
                try:
                    plan = json.loads(plan_str)
                except (json.JSONDecodeError, TypeError):
                    plan = {}

                past_tasks.append(
                    PastTask(
                        task_description=payload.get(
                            "task_description", ""
                        ),
                        execution_plan=plan,
                        step_results_summary=payload.get(
                            "step_results_summary", ""
                        ),
                        success=payload.get("success", True),
                        duration_seconds=payload.get(
                            "duration_seconds", 0.0
                        ),
                        tools_used=payload.get(
                            "tools_used", []
                        ),
                        timestamp=payload.get(
                            "timestamp", ""
                        ),
                        score=hit.score,
                        doc_id=str(hit.id),
                    )
                )

            return past_tasks

        except Exception as exc:
            logger.warning(
                "longterm_retrieve_failed",
                error=str(exc),
            )
            return []

    @property
    def document_count(self) -> int:
        """Number of documents in the main collection."""
        client = self._get_client()
        if client is None:
            return 0

        try:
            info = client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0
