"""Unit tests for LongTermMemory and Retriever."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orion.memory.embedder import Embedder
from orion.memory.longterm import LongTermMemory, PastTask
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
        mock_instance.encode.return_value = [[0.1, 0.2]]
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
        mock_instance.encode.return_value = [[0.1, 0.2]]
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
        # Mock returns 2 embeddings for the 2 new string
        mock_instance.encode.return_value = [[1.0], [2.0]]
        mock_st.return_value = mock_instance

        embedder = Embedder.get_instance()
        # Prefill cache for one string
        embedder._cache[embedder._hash("seen")] = [0.0]

        vectors = embedder.encode_batch(["seen", "new1", "new2"])

        assert len(vectors) == 3
        assert vectors[0] == [0.0]  # cached
        assert vectors[1] == [1.0]  # new
        assert vectors[2] == [2.0]  # new


class TestLongTermMemory:
    """Tests for LongTermMemory logic (mocked Qdrant)."""

    @patch("qdrant_client.QdrantClient")
    def test_lazy_connection(self, mock_qdrant: MagicMock) -> None:
        """Client connects only on first use."""
        memory = LongTermMemory()
        mock_qdrant.assert_not_called()

        memory._get_client()
        mock_qdrant.assert_called_once()

    @patch("orion.memory.longterm.LongTermMemory._get_client")
    def test_store_routing(self, mock_get_client: MagicMock) -> None:
        """Success/failure routes to appropriate collections."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        embedder = MagicMock()
        embedder.encode.return_value = [0.1]

        memory = LongTermMemory(embedder=embedder)

        # Success - routes to main collection
        memory.store("test", {}, "", success=True)
        call_args = mock_client.upsert.call_args[1]
        assert call_args["collection_name"] == "orion_tasks"

        # Failure - routes to failure collection
        memory.store("fail", {}, "", success=False)
        call_args = mock_client.upsert.call_args[1]
        assert call_args["collection_name"] == "orion_failures"

    @patch("orion.memory.longterm.LongTermMemory._get_client")
    def test_retrieve(self, mock_get_client: MagicMock) -> None:
        """Retrieve parses Qdrant points into PastTask."""
        mock_client = MagicMock()
        mock_point = MagicMock()
        mock_point.id = "123"
        mock_point.score = 0.95
        mock_point.payload = {
            "task_description": "Find files",
            "execution_plan": '{"steps": []}',
            "success": True,
        }
        mock_client.search.return_value = [mock_point]
        mock_get_client.return_value = mock_client

        embedder = MagicMock()
        embedder.encode.return_value = [0.1]

        memory = LongTermMemory(embedder=embedder)
        results = memory.retrieve("query")

        assert len(results) == 1
        assert results[0].doc_id == "123"
        assert results[0].task_description == "Find files"
        assert results[0].success is True


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
