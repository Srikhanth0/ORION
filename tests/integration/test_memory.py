"""Integration tests for Memory subsystem (Working + Embedder + ChromaDB).

Uses an in-memory ChromaDB client for speed — no disk I/O needed.
"""
from __future__ import annotations

import pytest

from orion.memory.embedder import Embedder
from orion.memory.longterm import LocalLongTermMemory
from orion.memory.working import WorkingMemory

try:
    import sentence_transformers  # noqa
except (ImportError, OSError):
    pytest.skip("sentence_transformers unavailable", allow_module_level=True)


@pytest.fixture(autouse=True)
def _reset_embedder() -> None:
    Embedder.reset()


class TestMemoryIntegration:
    """Integration test suite for Memory."""

    def test_working_memory_lifecycle(self) -> None:
        """Working memory manages adding, rendering, clear."""
        wm = WorkingMemory()
        wm.add_plan({"tasks": 1})
        wm.add_step_result("s1", "github", "done", True)
        wm.add_note("A standard note")

        ctx = wm.to_context_str()
        assert "[PLAN]" in ctx
        assert "[STEP_RESULT]" in ctx
        assert "[NOTE] A standard note" in ctx
        assert wm.total_tokens > 0

        wm.clear()
        assert wm.total_tokens == 0
        assert wm.to_context_str() == "No prior context."

    def test_embedder_caching_integration(self) -> None:
        """Embedder calculates embeddings and caches without mocks."""
        # This will download the small sentence-transformers model
        # on first run, which takes a few seconds but is required
        # for a true integration test.
        try:
            import sentence_transformers  # noqa
        except ImportError:
            pytest.skip("sentence_transformers required for tests.")

        embedder = Embedder.get_instance()
        vec1 = embedder.encode("Test string")

        assert len(vec1) == 384
        assert embedder.cache_size == 1

        # repeated
        vec2 = embedder.encode("Test string")
        assert vec1 == vec2
        assert embedder.cache_size == 1  # didn't grow

        # batch
        batch = embedder.encode_batch(["Test string", "Another string"])
        assert len(batch) == 2
        assert len(batch[0]) == 384
        assert len(batch[1]) == 384
        assert embedder.cache_size == 2

    def test_longterm_chromadb_in_memory(self) -> None:
        """LocalLongTermMemory works with in-memory ChromaDB client."""
        import chromadb

        # Use in-memory client instead of PersistentClient for speed
        in_memory_client = chromadb.Client()

        ltm = LocalLongTermMemory()
        ltm._client = in_memory_client
        ltm._ensure_collections()

        # Store a successful task
        doc_id = ltm.store(
            task_description="List all Python files",
            execution_plan={"steps": ["find *.py"]},
            step_results_summary="Found 42 files",
            success=True,
            duration_seconds=1.5,
            tools_used=["os_tools"],
        )
        assert len(doc_id) > 10  # uuid string
        assert ltm.document_count == 1

        # Store a failed task (should go to failure collection)
        ltm.store(
            task_description="Delete system32",
            execution_plan={},
            step_results_summary="Permission denied",
            success=False,
        )
        # Main collection still has 1
        assert ltm.document_count == 1

    def test_longterm_roundtrip(self) -> None:
        """Store a task, retrieve it, verify fields match."""
        import chromadb

        in_memory_client = chromadb.Client()

        ltm = LocalLongTermMemory()
        ltm._client = in_memory_client
        ltm._ensure_collections()

        ltm.store(
            task_description="Clone the ORION repository",
            execution_plan={"steps": ["git clone"]},
            step_results_summary="Cloned successfully",
            success=True,
            duration_seconds=3.2,
            tools_used=["github_tools"],
        )

        # Retrieve with a similar query
        results = ltm.retrieve(
            "clone a git repo",
            top_k=3,
            score_threshold=0.0,  # accept any score for test
        )

        assert len(results) >= 1
        task = results[0]
        assert task.task_description == "Clone the ORION repository"
        assert task.success is True
        assert "github_tools" in task.tools_used

    def test_longterm_clear(self) -> None:
        """Clear empties all collections."""
        import chromadb

        in_memory_client = chromadb.Client()

        # Use unique collection names to avoid leaking state
        # from other test methods
        ltm = LocalLongTermMemory(
            collection="clear_test_tasks",
        )
        ltm._client = in_memory_client
        ltm._collection_name = "clear_test_tasks"
        ltm._failure_collection_name = "clear_test_failures"
        ltm._ensure_collections()

        ltm.store("task 1", {}, "done", True)
        assert ltm.document_count == 1

        ltm.clear()
        assert ltm.document_count == 0
