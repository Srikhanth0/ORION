"""Execution result containers for ORION.

Defines the data structures produced by the Executor and Verifier
agents to report on subtask outcomes, rollback checkpoints, and
aggregate task results.

Module Contract
---------------
- **Inputs**: Executor step outputs, Verifier assertions.
- **Outputs**: ``StepResult``, ``TaskResult``, ``RollbackPoint`` —
  frozen Pydantic models (pure data containers).
- **Failure Modes**: None. These are value types with no logic.
- **Must NOT Know About**: Agent implementations, tool internals,
  LLM providers, or orchestrator logic.

Depends On
----------
- ``pydantic`` for data validation.
- ``orion.core.task.TaskStatus`` for status enum.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from orion.core.task import TaskStatus


class StepResult(BaseModel, frozen=True):
    """Result of a single subtask execution.

    Produced by the Executor agent after invoking a tool. The
    Verifier agent consumes a list of these to run assertions.

    Args:
        subtask_id: ID of the subtask that was executed.
        ok: Whether the execution succeeded.
        output: The tool's output payload on success.
        error: Error description on failure.
        duration_ms: Wall-clock execution time in milliseconds.
        provider_used: Which LLM provider was used for reasoning
            during this step (for cost tracking).
        attempt: Which retry attempt produced this result (1-indexed).
    """

    subtask_id: str
    ok: bool
    output: object = None
    error: str | None = None
    duration_ms: int = 0
    provider_used: str | None = None
    attempt: int = 1


class RollbackPoint(BaseModel, frozen=True):
    """Checkpoint created before an Executor step for safe rollback.

    The RollbackEngine creates these before each potentially
    destructive operation. If the Supervisor decides to rollback,
    checkpoints are replayed in reverse order.

    Args:
        subtask_id: The subtask this checkpoint was created for.
        checkpoint_type: Category of checkpoint (file, git_stash,
            api_call_marker, process_state).
        checkpoint_data: Serialized checkpoint data (e.g., file hash
            + content, git stash ref, working directory path).
        created_at: UTC timestamp of checkpoint creation.
    """

    subtask_id: str
    checkpoint_type: str
    checkpoint_data: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskResult(BaseModel, frozen=True):
    """Aggregate result of a complete task execution.

    Produced by the orchestrator after all agents (Planner, Executor,
    Verifier, Supervisor) have finished processing.

    Args:
        task_id: ID of the completed task.
        status: Final task status (COMPLETED, FAILED, ROLLED_BACK).
        step_results: Results from each subtask execution.
        rollback_points: Checkpoints available for rollback.
        total_duration_ms: Total wall-clock time for the task.
        total_cost_usd: Estimated cost in USD across all LLM providers.
        verification_passed: Whether the Verifier approved all results.
        verification_notes: Verifier's per-step notes or critique.
    """

    task_id: str
    status: TaskStatus
    step_results: list[StepResult] = Field(default_factory=list)
    rollback_points: list[RollbackPoint] = Field(default_factory=list)
    total_duration_ms: int = 0
    total_cost_usd: float = 0.0
    verification_passed: bool = False
    verification_notes: str = ""
