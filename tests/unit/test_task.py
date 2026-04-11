"""Unit tests for orion.core.task — Task, Subtask, TaskDAG.

Tests cover:
- Subtask creation with defaults and explicit values
- TaskDAG topological ordering
- TaskDAG cycle detection (raises TaskValidationError)
- TaskDAG missing dependency detection
- TaskDAG duplicate ID detection
- Task creation and status lifecycle
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from orion.core.exceptions import TaskValidationError
from orion.core.task import Subtask, Task, TaskDAG, TaskStatus


class TestSubtask:
    """Tests for the Subtask frozen model."""

    def test_subtask_creation_defaults(self) -> None:
        """Subtask with only required fields gets sensible defaults."""
        s = Subtask(action="do something")
        assert s.action == "do something"
        assert s.id.startswith("s_")
        assert s.tool_hint == ""
        assert s.depends_on == []
        assert s.params == {}
        assert s.timeout_seconds == 30

    def test_subtask_creation_explicit(self) -> None:
        """Subtask with all fields explicitly set."""
        s = Subtask(
            id="custom_id",
            action="clone repo",
            tool_hint="github_clone",
            depends_on=["prev"],
            params={"url": "https://github.com/org/repo"},
            timeout_seconds=60,
        )
        assert s.id == "custom_id"
        assert s.tool_hint == "github_clone"
        assert s.depends_on == ["prev"]
        assert s.params["url"] == "https://github.com/org/repo"
        assert s.timeout_seconds == 60

    def test_subtask_is_frozen(self) -> None:
        """Subtask fields cannot be mutated after construction."""
        s = Subtask(action="do something")
        with pytest.raises(ValidationError):  # pydantic frozen model
            s.action = "changed"  # type: ignore[misc]


class TestTaskDAG:
    """Tests for the TaskDAG model and its topological sort."""

    def test_dag_topological_order_linear(self, sample_dag: TaskDAG) -> None:
        """Linear DAG s1 → s2 → s3 returns correct order."""
        order = sample_dag.topological_order()
        ids = [s.id for s in order]
        assert ids == ["s1", "s2", "s3"]

    def test_dag_topological_order_diamond(self) -> None:
        """Diamond DAG: s1 → {s2, s3} → s4."""
        dag = TaskDAG(
            task_id="t_diamond",
            instruction="diamond shape",
            subtasks=[
                Subtask(id="s1", action="root"),
                Subtask(id="s2", action="left", depends_on=["s1"]),
                Subtask(id="s3", action="right", depends_on=["s1"]),
                Subtask(id="s4", action="join", depends_on=["s2", "s3"]),
            ],
        )
        order = dag.topological_order()
        ids = [s.id for s in order]

        # s1 must come first, s4 must come last
        assert ids[0] == "s1"
        assert ids[-1] == "s4"
        # s2 and s3 can be in either order, but both before s4
        assert set(ids[1:3]) == {"s2", "s3"}

    def test_dag_independent_subtasks(self) -> None:
        """DAG with no dependencies: all subtasks are roots."""
        dag = TaskDAG(
            task_id="t_parallel",
            instruction="all parallel",
            subtasks=[
                Subtask(id="s1", action="task A"),
                Subtask(id="s2", action="task B"),
                Subtask(id="s3", action="task C"),
            ],
        )
        order = dag.topological_order()
        assert len(order) == 3

    def test_dag_cycle_detection_self_loop(self) -> None:
        """Subtask depending on itself raises TaskValidationError."""
        with pytest.raises(TaskValidationError, match="cycle"):
            TaskDAG(
                task_id="t_cycle",
                instruction="whoops",
                subtasks=[
                    Subtask(id="s1", action="loop", depends_on=["s1"]),
                ],
            )

    def test_dag_cycle_detection_mutual(self) -> None:
        """Two subtasks forming a mutual cycle raises TaskValidationError."""
        with pytest.raises(TaskValidationError, match="cycle"):
            TaskDAG(
                task_id="t_mutual",
                instruction="mutual deps",
                subtasks=[
                    Subtask(id="s1", action="A", depends_on=["s2"]),
                    Subtask(id="s2", action="B", depends_on=["s1"]),
                ],
            )

    def test_dag_cycle_detection_three_node(self) -> None:
        """Three-node cycle: s1 → s2 → s3 → s1."""
        with pytest.raises(TaskValidationError, match="cycle"):
            TaskDAG(
                task_id="t_triangle",
                instruction="three-node cycle",
                subtasks=[
                    Subtask(id="s1", action="A", depends_on=["s3"]),
                    Subtask(id="s2", action="B", depends_on=["s1"]),
                    Subtask(id="s3", action="C", depends_on=["s2"]),
                ],
            )

    def test_dag_missing_dependency(self) -> None:
        """Reference to non-existent subtask ID raises TaskValidationError."""
        with pytest.raises(TaskValidationError, match="unknown subtask"):
            TaskDAG(
                task_id="t_missing",
                instruction="missing dep",
                subtasks=[
                    Subtask(id="s1", action="A", depends_on=["s_nonexistent"]),
                ],
            )

    def test_dag_duplicate_id(self) -> None:
        """Duplicate subtask IDs raise TaskValidationError."""
        with pytest.raises(TaskValidationError, match="Duplicate"):
            TaskDAG(
                task_id="t_dup",
                instruction="duplicate ids",
                subtasks=[
                    Subtask(id="s1", action="A"),
                    Subtask(id="s1", action="B"),
                ],
            )

    def test_dag_empty_subtasks(self) -> None:
        """Empty subtask list raises TaskValidationError."""
        with pytest.raises(TaskValidationError, match="at least one"):
            TaskDAG(
                task_id="t_empty",
                instruction="nothing to do",
                subtasks=[],
            )

    def test_dag_get_subtask(self, sample_dag: TaskDAG) -> None:
        """get_subtask returns the correct subtask by ID."""
        s = sample_dag.get_subtask("s2")
        assert s.id == "s2"
        assert s.action == "Run tests"

    def test_dag_get_subtask_not_found(self, sample_dag: TaskDAG) -> None:
        """get_subtask raises TaskValidationError for unknown ID."""
        with pytest.raises(TaskValidationError, match="not found"):
            sample_dag.get_subtask("s999")


class TestTask:
    """Tests for the Task lifecycle model."""

    def test_task_creation_defaults(self) -> None:
        """Task with only required field gets defaults."""
        t = Task(instruction="do the thing")
        assert t.instruction == "do the thing"
        assert t.task_id.startswith("t_")
        assert t.status == TaskStatus.PENDING
        assert t.dag is None
        assert t.metadata == {}
        assert t.created_at is not None

    def test_task_with_dag(self, sample_dag: TaskDAG) -> None:
        """Task can hold a TaskDAG reference."""
        t = Task(
            task_id="t_with_dag",
            instruction="planned task",
            dag=sample_dag,
            status=TaskStatus.PLANNING,
        )
        assert t.dag is not None
        assert len(t.dag.subtasks) == 3

    def test_task_status_is_mutable(self) -> None:
        """Task status can be updated (unlike frozen models)."""
        t = Task(instruction="mutable")
        t.status = TaskStatus.EXECUTING
        assert t.status == TaskStatus.EXECUTING

    def test_task_status_enum_values(self) -> None:
        """All expected TaskStatus values exist."""
        expected = {
            "pending", "planning", "executing", "verifying",
            "completed", "failed", "rolled_back",
        }
        actual = {s.value for s in TaskStatus}
        assert actual == expected
