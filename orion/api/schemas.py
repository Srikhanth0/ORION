"""Pydantic request/response schemas for the ORION API.

All schemas use ``model_config = ConfigDict(frozen=True)``
for immutability. Error responses follow RFC 7807.
"""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ── Enums ─────────────────────────────────────────────────


class TaskStatus(StrEnum):
    """Task lifecycle status."""

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class EventType(StrEnum):
    """SSE event types."""

    STEP_START = "STEP_START"
    STEP_DONE = "STEP_DONE"
    STEP_FAILED = "STEP_FAILED"
    AGENT_THOUGHT = "AGENT_THOUGHT"
    TASK_DONE = "TASK_DONE"

    # Fine-grained pipeline events (Phase 5)
    PLANNER_START = "planner_start"
    SUBTASK_QUEUED = "subtask_queued"
    SUBTASK_START = "subtask_start"
    SUBTASK_RESULT = "subtask_result"
    VERIFIER_RESULT = "verifier_result"
    SUPERVISOR_DECISION = "supervisor_decision"
    DONE = "done"


# ── Requests ──────────────────────────────────────────────


class TaskRequest(BaseModel):
    """POST /v1/tasks request body.

    Attributes:
        instruction: Natural language task.
        context: Optional OS context override.
        timeout_seconds: Task timeout.
        hitl_webhook_url: URL for HITL approval callbacks.
    """

    model_config = ConfigDict(frozen=True)

    instruction: str = Field(
        ..., min_length=1, max_length=5000
    )
    context: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = Field(
        default=300, ge=10, le=3600
    )
    hitl_webhook_url: str | None = None
    safe_mode: bool = False


# ── Responses ─────────────────────────────────────────────


class TaskResponse(BaseModel):
    """Minimal response after task submission.

    Attributes:
        task_id: Assigned task ID.
        status: Current status.
        created_at: Creation timestamp.
        estimated_seconds: Rough estimate.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: TaskStatus
    created_at: datetime
    estimated_seconds: int | None = None


class StepSummary(BaseModel):
    """Summary of a single execution step."""

    model_config = ConfigDict(frozen=True)

    subtask_id: str
    tool: str = ""
    ok: bool = True
    output: str = ""
    duration_ms: float = 0.0


class TaskDetailResponse(BaseModel):
    """GET /v1/tasks/{task_id} full detail response.

    Attributes:
        task_id: Task ID.
        status: Current status.
        created_at: Creation timestamp.
        result: Final task result (if completed).
        steps: List of step summaries.
        cost_usd: Estimated cost.
        trace_url: LangSmith trace link.
    """

    model_config = ConfigDict(frozen=True)

    task_id: str
    status: TaskStatus
    created_at: datetime
    result: dict[str, Any] | None = None
    steps: list[StepSummary] = Field(default_factory=list)
    cost_usd: float | None = None
    trace_url: str | None = None


class TaskEvent(BaseModel):
    """SSE event payload.

    Attributes:
        event_type: Event classification.
        data: Event data.
        timestamp: Event timestamp.
    """

    model_config = ConfigDict(frozen=True)

    event_type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime


class ToolSchema(BaseModel):
    """Tool description for GET /v1/tools.

    Attributes:
        name: Tool action name.
        description: What the tool does.
        category: Tool category.
        is_destructive: Whether it mutates state.
        params_schema: JSON Schema for parameters.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    category: str = ""
    is_destructive: bool = False
    params_schema: dict[str, Any] = Field(
        default_factory=dict
    )


# ── Error Responses (RFC 7807) ───────────────────────────


class ProblemDetail(BaseModel):
    """RFC 7807 Problem Details error response.

    Attributes:
        type: Problem type URI.
        title: Short human-readable title.
        status: HTTP status code.
        detail: Detailed description.
        task_id: Related task ID (optional).
    """

    model_config = ConfigDict(frozen=True)

    type: str = "about:blank"
    title: str = "Error"
    status: int = 500
    detail: str = ""
    task_id: str | None = None
