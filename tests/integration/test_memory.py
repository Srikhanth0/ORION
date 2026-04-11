"""Integration tests for Memory subsystem (Working + Embedder).

LongTermMemory integration requiring live Qdrant is skipped
unless explicitly requested via env var.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orion.memory.embedder import Embedder
from orion.memory.longterm import LongTermMemory
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

    def test_longterm_graceful_degradation(self) -> None:
        """LongTermMemory falls back gracefully if Qdrant isn't running."""
        # Provide real Embedder but point Qdrant to an invalid port
        # so connection fails immediately.
        ltm = LongTermMemory(url="http://localhost:11111")
        # Overwrite _ensure_collections so it doesn't hard-crash
        ltm._ensure_collections = MagicMock()

        # Store should catch exception and return a random doc_id
        doc_id = ltm.store("desc", {}, "sum", True)
        assert len(doc_id) > 10  # uuid string

        # Retrieve should catch exception and return empty list
        results = ltm.retrieve("query")
        assert results == []
        assert ltm.document_count == 0
