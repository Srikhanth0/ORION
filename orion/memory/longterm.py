"""LocalLongTermMemory — persistent cross-task memory backed by ChromaDB.

Stores successful task results with embeddings for semantic retrieval.
Failed tasks go to a separate failure collection.

Module Contract
---------------
- **Inputs**: TaskResult objects.
- **Outputs**: PastTask retrieval via cosine similarity.

Depends On
----------
- ``chromadb`` (PersistentClient)
- ``orion.memory.embedder`` (Embedder)
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

from orion.memory.embedder import Embedder

logger = structlog.get_logger(__name__)

_DEFAULT_COLLECTION = "orion_tasks"
_FAILURE_COLLECTION = "orion_failures"
_DEFAULT_PERSIST_PATH = ".orion_memory"


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


class LocalLongTermMemory:
    """Persistent cross-task memory backed by embedded ChromaDB.

    Stores successful tasks in the main collection and
    failures in a separate collection. Provides semantic
    retrieval for few-shot examples.

    Args:
        persist_path: Directory for ChromaDB storage.
        collection: Main collection name.
        embedder: Embedder instance.
    """

    def __init__(
        self,
        persist_path: str | None = None,
        collection: str = _DEFAULT_COLLECTION,
        embedder: Embedder | None = None,
    ) -> None:
        self._persist_path = persist_path or os.environ.get(
            "CHROMA_PERSIST_PATH", _DEFAULT_PERSIST_PATH
        )
        self._collection_name = collection
        self._failure_collection_name = _FAILURE_COLLECTION
        self._embedder = embedder or Embedder.get_instance()
        self._client: Any = None
        self._collection: Any = None
        self._failure_collection: Any = None
        self._fallback: dict[str, list[PastTask]] = {}

    def _get_client(self) -> Any:
        """Lazy-load the ChromaDB client."""
        if self._client is not None:
            return self._client

        try:
            import chromadb

            self._client = chromadb.PersistentClient(
                path=self._persist_path
            )
            self._ensure_collections()
            logger.info(
                "chromadb_connected",
                persist_path=self._persist_path,
                collection=self._collection_name,
            )
        except Exception as exc:
            logger.warning(
                "chromadb_connection_failed",
                persist_path=self._persist_path,
                error=str(exc),
            )
            self._client = None

        return self._client

    def _ensure_collections(self) -> None:
        """Create or get collections."""
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._failure_collection = self._client.get_or_create_collection(
            name=self._failure_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "collections_ready",
            main=self._collection_name,
            failure=self._failure_collection_name,
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

        metadata = {
            "task_description": task_description,
            "execution_plan": json.dumps(execution_plan),
            "step_results_summary": step_results_summary,
            "success": success,
            "duration_seconds": duration_seconds,
            "tools_used": json.dumps(tools_used or []),
            "timestamp": datetime.now(
                tz=UTC
            ).isoformat(),
            "agent_versions": json.dumps(
                agent_versions or {}
            ),
        }

        client = self._get_client()
        if client is None:
            logger.warning(
                "longterm_store_skipped",
                reason="no_chromadb_connection",
            )
            # ORION-FIX: Graceful degradation when memory vector DB is unavailable
            self._fallback.setdefault('fallback_key', []).append(PastTask(
                task_description=task_description,
                execution_plan=execution_plan,
                step_results_summary=step_results_summary,
                success=success,
                duration_seconds=duration_seconds,
                tools_used=tools_used or [],
                doc_id=doc_id,
            ))
            return doc_id

        try:
            target = (
                self._collection if success
                else self._failure_collection
            )

            target.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[task_description],
            )

            logger.info(
                "longterm_stored",
                doc_id=doc_id,
                collection=target.name,
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
            # naive fallback: return last top_k stored items
            all_items = [p for v in self._fallback.values() for p in v]
            return all_items[-top_k:] if all_items else []

        try:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                include=["metadatas", "distances"],
            )

            past_tasks: list[PastTask] = []

            if not results or not results.get("ids"):
                return []

            ids = results["ids"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]

            for i, doc_id in enumerate(ids):
                metadata = metadatas[i] if metadatas else {}
                # ChromaDB returns distances; for cosine,
                # similarity = 1 - distance
                distance = distances[i] if distances else 1.0
                score = 1.0 - distance

                if score < score_threshold:
                    continue

                plan_str = metadata.get(
                    "execution_plan", "{}"
                )
                try:
                    plan = json.loads(plan_str)
                except (json.JSONDecodeError, TypeError):
                    plan = {}

                tools_str = metadata.get("tools_used", "[]")
                try:
                    tools = json.loads(tools_str)
                except (json.JSONDecodeError, TypeError):
                    tools = []

                past_tasks.append(
                    PastTask(
                        task_description=metadata.get(
                            "task_description", ""
                        ),
                        execution_plan=plan,
                        step_results_summary=metadata.get(
                            "step_results_summary", ""
                        ),
                        success=metadata.get("success", True),
                        duration_seconds=metadata.get(
                            "duration_seconds", 0.0
                        ),
                        tools_used=tools,
                        timestamp=metadata.get(
                            "timestamp", ""
                        ),
                        score=score,
                        doc_id=str(doc_id),
                    )
                )

            return past_tasks

        except Exception as exc:
            logger.warning(
                "longterm_retrieve_failed",
                error=str(exc),
            )
            return []

    def clear(self) -> None:
        """Delete and recreate all collections."""
        client = self._get_client()
        if client is None:
            return

        try:
            client.delete_collection(self._collection_name)
            client.delete_collection(
                self._failure_collection_name
            )
            self._collection = None
            self._failure_collection = None
            self._ensure_collections()
            logger.info("longterm_memory_cleared")
        except Exception as exc:
            logger.warning(
                "longterm_clear_failed",
                error=str(exc),
            )

    @property
    def document_count(self) -> int:
        """Number of documents in the main collection."""
        client = self._get_client()
        if client is None:
            return 0

        try:
            return self._collection.count()
        except Exception:
            return 0


# Backward-compatible alias
LongTermMemory = LocalLongTermMemory
