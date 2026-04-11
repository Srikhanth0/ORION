"""Unit tests for orion.core.result — StepResult, TaskResult, RollbackPoint.

Tests cover:
- StepResult creation for success and failure cases
- RollbackPoint creation and timestamp generation
- TaskResult aggregation with mixed step outcomes
- Frozen model immutability
- Serialization round-trip (model_dump / model_validate)
"""
from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from orion.core.result import RollbackPoint, StepResult, TaskResult
from orion.core.task import TaskStatus


class TestStepResult:
    """Tests for the StepResult frozen model."""

    def test_step_result_success(
        self, sample_step_result_success: StepResult
    ) -> None:
        """Successful StepResult has ok=True and output set."""
        r = sample_step_result_success
        assert r.ok is True
        assert r.error is None
        assert r.subtask_id == "s1"
        assert r.duration_ms == 1200
        assert r.provider_used == "groq"
        assert r.attempt == 1
        assert isinstance(r.output, dict)

    def test_step_result_failure(
        self, sample_step_result_failure: StepResult
    ) -> None:
        """Failed StepResult has ok=False and error message."""
        r = sample_step_result_failure
        assert r.ok is False
        assert r.error is not None
        assert "exit code 1" in r.error
        assert r.attempt == 2

    def test_step_result_defaults(self) -> None:
        """StepResult with minimal fields gets sensible defaults."""
        r = StepResult(subtask_id="s1", ok=True)
        assert r.output is None
        assert r.error is None
        assert r.duration_ms == 0
        assert r.provider_used is None
        assert r.attempt == 1

    def test_step_result_is_frozen(self) -> None:
        """StepResult fields cannot be mutated."""
        r = StepResult(subtask_id="s1", ok=True)
        with pytest.raises(ValidationError):
            r.ok = False  # type: ignore[misc]

    def test_step_result_serialization(self) -> None:
        """StepResult round-trips through model_dump / model_validate."""
        original = StepResult(
            subtask_id="s1",
            ok=True,
            output={"key": "value"},
            duration_ms=500,
            provider_used="vllm",
        )
        data = original.model_dump()
        restored = StepResult.model_validate(data)
        assert restored.subtask_id == original.subtask_id
        assert restored.ok == original.ok
        assert restored.output == original.output


class TestRollbackPoint:
    """Tests for the RollbackPoint frozen model."""

    def test_rollback_point_creation(
        self, sample_rollback_point: RollbackPoint
    ) -> None:
        """RollbackPoint stores checkpoint data correctly."""
        rp = sample_rollback_point
        assert rp.subtask_id == "s1"
        assert rp.checkpoint_type == "file"
        assert rp.checkpoint_data["path"] == "/workspace/repo"
        assert rp.checkpoint_data["hash"] == "abc123"

    def test_rollback_point_auto_timestamp(self) -> None:
        """RollbackPoint gets an auto-generated UTC timestamp."""
        rp = RollbackPoint(subtask_id="s1", checkpoint_type="git_stash")
        assert rp.created_at is not None
        assert rp.created_at.tzinfo == UTC

    def test_rollback_point_is_frozen(self) -> None:
        """RollbackPoint fields cannot be mutated."""
        rp = RollbackPoint(subtask_id="s1", checkpoint_type="file")
        with pytest.raises(ValidationError):
            rp.checkpoint_type = "changed"  # type: ignore[misc]

    def test_rollback_point_serialization(self) -> None:
        """RollbackPoint round-trips through model_dump / model_validate."""
        original = RollbackPoint(
            subtask_id="s1",
            checkpoint_type="file",
            checkpoint_data={"path": "/tmp/x"},
        )
        data = original.model_dump()
        restored = RollbackPoint.model_validate(data)
        assert restored.subtask_id == original.subtask_id
        assert restored.checkpoint_type == original.checkpoint_type


class TestTaskResult:
    """Tests for the TaskResult aggregate model."""

    def test_task_result_aggregation(
        self, sample_task_result: TaskResult
    ) -> None:
        """TaskResult aggregates multiple StepResults."""
        tr = sample_task_result
        assert tr.task_id == "t_test001"
        assert tr.status == TaskStatus.FAILED
        assert len(tr.step_results) == 2
        assert tr.total_duration_ms == 6600
        assert tr.total_cost_usd == pytest.approx(0.003)
        assert tr.verification_passed is False
        assert "s2 failed" in tr.verification_notes

    def test_task_result_empty_steps(self) -> None:
        """TaskResult can be created with no step results."""
        tr = TaskResult(task_id="t_empty", status=TaskStatus.COMPLETED)
        assert tr.step_results == []
        assert tr.rollback_points == []
        assert tr.total_duration_ms == 0
        assert tr.total_cost_usd == 0.0

    def test_task_result_with_rollback_points(
        self, sample_rollback_point: RollbackPoint
    ) -> None:
        """TaskResult stores rollback points for the Supervisor."""
        tr = TaskResult(
            task_id="t_rb",
            status=TaskStatus.ROLLED_BACK,
            rollback_points=[sample_rollback_point],
        )
        assert len(tr.rollback_points) == 1
        assert tr.rollback_points[0].checkpoint_type == "file"

    def test_task_result_is_frozen(self) -> None:
        """TaskResult fields cannot be mutated."""
        tr = TaskResult(task_id="t1", status=TaskStatus.COMPLETED)
        with pytest.raises(ValidationError):
            tr.status = TaskStatus.FAILED  # type: ignore[misc]

    def test_task_result_serialization(self) -> None:
        """TaskResult round-trips through JSON serialization."""
        tr = TaskResult(
            task_id="t_serial",
            status=TaskStatus.COMPLETED,
            step_results=[StepResult(subtask_id="s1", ok=True)],
            total_duration_ms=100,
        )
        json_str = tr.model_dump_json()
        restored = TaskResult.model_validate_json(json_str)
        assert restored.task_id == "t_serial"
        assert len(restored.step_results) == 1
