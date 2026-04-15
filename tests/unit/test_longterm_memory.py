"""Unit tests for LocalLongTermMemory and Retriever."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orion.memory.embedder import Embedder
from orion.memory.longterm import LocalLongTermMemory, LongTermMemory, PastTask
from orion.memory.retriever import MemoryRetriever
from orion.memory.working import WorkingMemory

try:
    import sentence_transformers  # noqa
except (ImportError, OSError):
    pytest.skip("sentence_transformers unavailable", allow_module_level=True)


class TestEmbedder:
    """Tests for Embedder singleton."""

    @pytest.fixture(autouse=True)
    def _reset_embedder(self) -> None:
        Embedder.reset()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_singleton(self, mock_st: MagicMock) -> None:
        """Embedder loads model once and returns mock vector."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2]])
        mock_st.return_value = mock_instance

        embedder = Embedder.get_instance()
        vec = embedder.encode("test string")

        assert len(vec) == 2
        mock_st.assert_called_once()
        mock_instance.encode.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_caching(self, mock_st: MagicMock) -> None:
        """Repeated encode requests use cache."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = np.array([[0.1, 0.2]])
        mock_st.return_value = mock_instance

        embedder = Embedder.get_instance()
        embedder.encode("cached")
        embedder.encode("cached")

        # Encode hit model only once
        mock_instance.encode.assert_called_once()
        assert embedder.cache_size == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_encode_batch(self, mock_st: MagicMock) -> None:
        """encode_batch handles a mix of cached and uncached texts."""
        mock_instance = MagicMock()
        # Mock returns 2 embeddings for the 2 new strings
        mock_instance.encode.return_value = np.array([[1.0], [2.0]])
        mock_st.return_value = mock_instance

        embedder = Embedder.get_instance()
        # Prefill cache for one string
        embedder._cache[embedder._hash("seen")] = [0.0]

        vectors = embedder.encode_batch(["seen", "new1", "new2"])

        assert len(vectors) == 3
        assert vectors[0] == [0.0]  # cached
        assert vectors[1] == [1.0]  # new
        assert vectors[2] == [2.0]  # new


class TestLocalLongTermMemory:
    """Tests for LocalLongTermMemory logic (mocked ChromaDB)."""

    @patch("chromadb.PersistentClient")
    def test_lazy_connection(self, mock_chromadb: MagicMock) -> None:
        """Client connects only on first use."""
        memory = LocalLongTermMemory()
        mock_chromadb.assert_not_called()

        memory._get_client()
        mock_chromadb.assert_called_once()

    @patch("orion.memory.longterm.LocalLongTermMemory._get_client")
    def test_store_routing(self, mock_get_client: MagicMock) -> None:
        """Success/failure routes to appropriate collections."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_failure_collection = MagicMock()
        mock_get_client.return_value = mock_client

        embedder = MagicMock()
        embedder.encode.return_value = [0.1]

        memory = LocalLongTermMemory(embedder=embedder)
        memory._collection = mock_collection
        memory._failure_collection = mock_failure_collection

        # Success - routes to main collection
        memory.store("test", {}, "", success=True)
        mock_collection.upsert.assert_called_once()
        mock_failure_collection.upsert.assert_not_called()

        # Reset mocks
        mock_collection.reset_mock()
        mock_failure_collection.reset_mock()

        # Failure - routes to failure collection
        memory.store("fail", {}, "", success=False)
        mock_failure_collection.upsert.assert_called_once()
        mock_collection.upsert.assert_not_called()

    @patch("orion.memory.longterm.LocalLongTermMemory._get_client")
    def test_retrieve(self, mock_get_client: MagicMock) -> None:
        """Retrieve parses ChromaDB results into PastTask."""
        mock_client = MagicMock()
        mock_collection = MagicMock()

        # ChromaDB query returns nested lists
        mock_collection.query.return_value = {
            "ids": [["123"]],
            "metadatas": [[{
                "task_description": "Find files",
                "execution_plan": '{"steps": []}',
                "success": True,
                "tools_used": "[]",
            }]],
            "distances": [[0.05]],  # cosine distance
        }
        mock_get_client.return_value = mock_client

        embedder = MagicMock()
        embedder.encode.return_value = [0.1]

        memory = LocalLongTermMemory(embedder=embedder)
        memory._collection = mock_collection
        results = memory.retrieve("query")

        assert len(results) == 1
        assert results[0].doc_id == "123"
        assert results[0].task_description == "Find files"
        assert results[0].success is True
        assert results[0].score == pytest.approx(0.95)

    @patch("orion.memory.longterm.LocalLongTermMemory._get_client")
    def test_clear(self, mock_get_client: MagicMock) -> None:
        """Clear deletes and recreates collections."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        memory = LocalLongTermMemory()
        memory._collection = MagicMock()
        memory._failure_collection = MagicMock()
        memory._client = mock_client

        memory.clear()

        # Should delete both collections
        assert mock_client.delete_collection.call_count == 2
        # Should recreate via get_or_create_collection
        assert mock_client.get_or_create_collection.call_count == 2

    def test_backward_compatible_alias(self) -> None:
        """LongTermMemory alias points to LocalLongTermMemory."""
        assert LongTermMemory is LocalLongTermMemory


class TestMemoryRetriever:
    """Tests for MemoryRetriever facade."""

    def test_get_context_combined(self) -> None:
        """Retriever combines working and longterm memory."""
        working = WorkingMemory()
        working.add_note("Working note")

        longterm = MagicMock()
        longterm.retrieve.return_value = [
            PastTask(
                task_description="Old task",
                success=True,
                tools_used=[],
            )
        ]

        retriever = MemoryRetriever(working=working, longterm=longterm)
        ctx = retriever.get_context(query="test")

        assert "=== WORKING MEMORY ===" in ctx
        assert "[NOTE] Working note" in ctx
        assert "=== PAST TASKS ===" in ctx
        assert "Old task" in ctx
