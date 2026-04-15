"""Unit tests for WorkingMemory."""

from __future__ import annotations

import pytest

from orion.memory.working import WorkingMemory

try:
    import sentence_transformers  # noqa
except (ImportError, OSError):
    pytest.skip("sentence_transformers unavailable", allow_module_level=True)


class TestWorkingMemory:
    """Tests for WorkingMemory."""

    def test_add_plan(self) -> None:
        """Adding a plan creates a plan entry and increments tokens."""
        memory = WorkingMemory()
        plan = {"steps": ["step1", "step2"]}

        memory.add_plan(plan)

        assert memory.entry_count == 1
        assert memory._entries[0].role == "plan"
        assert memory._entries[0].metadata["type"] == "plan"
        assert "step1" in memory._entries[0].content
        assert memory.total_tokens > 0

    def test_add_step_result(self) -> None:
        """Adding step result creates a step_result entry."""
        memory = WorkingMemory()

        memory.add_step_result(
            subtask_id="s1",
            tool="SHELL",
            output="Done",
            success=True,
        )

        assert memory.entry_count == 1
        assert memory._entries[0].role == "step_result"
        assert "OK" in memory._entries[0].content
        assert "s1" in memory._entries[0].content

    def test_add_note(self) -> None:
        """Adding a note creates a note entry."""
        memory = WorkingMemory()

        memory.add_note("This is a temp note", agent="Planner")

        assert memory.entry_count == 1
        assert memory._entries[0].role == "note"
        assert memory._entries[0].content == "This is a temp note"
        assert memory._entries[0].metadata["agent"] == "Planner"

    def test_to_context_str(self) -> None:
        """to_context_str renders format correctly."""
        memory = WorkingMemory()
        memory.add_note("Note 1")
        memory.add_step_result("s1", "T", "out", True)

        ctx = memory.to_context_str()
        assert "[NOTE]" in ctx
        assert "[STEP_RESULT]" in ctx
        assert "---" in ctx

    def test_clear(self) -> None:
        """Clear empties memory and resets tokens."""
        memory = WorkingMemory()
        memory.add_note("Note")

        assert memory.entry_count == 1
        assert memory.total_tokens > 0

        memory.clear()

        assert memory.entry_count == 0
        assert memory.total_tokens == 0

    def test_utilization(self) -> None:
        """Utilization calculates fraction correctly."""
        memory = WorkingMemory(max_tokens=100)
        assert memory.utilization == 0.0

        memory.add_note("A" * 200)  # ~ 200 // 4 = 50 tokens
        assert memory.utilization > 0.0

    def test_eviction(self) -> None:
        """Exceeding 80% utilization triggers eviction."""
        # 40 tokens max
        memory = WorkingMemory(max_tokens=40)

        # Add 3 notes, ~12 tokens each
        memory.add_note("1" * 48)
        memory.add_note("2" * 48)

        # Should be below 80% (24 / 40)
        assert memory.entry_count == 2
        assert memory._entries[0].role == "note"

        # Third note pushes over 32 tokens (80%)
        memory.add_note("3" * 48)

        # The new state should contain a summary and the newest entry
        assert memory._entries[0].role == "summary"
        assert "note" in memory._entries[1].role
