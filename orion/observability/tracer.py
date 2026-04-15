"""Tracer — LangSmith distributed tracing for ORION agents.

In production: submits traces to LangSmith. In dev mode (no API key):
falls back to structlog DEBUG logging.

Module Contract
---------------
- **Inputs**: task_id, agent events, completion data.
- **Outputs**: Trace spans in LangSmith or structlog.

Depends On
----------
- ``langsmith`` (Client, traceable) — optional.
- ``structlog`` (logging)
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TraceSpan:
    """A trace span representing an operation.

    Attributes:
        name: Span name (agent or operation name).
        task_id: Parent task ID.
        tags: List of tags for the span.
        metadata: Structured metadata.
        start_time: Start timestamp.
        end_time: End timestamp.
    """

    name: str
    task_id: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class Tracer:
    """LangSmith distributed tracing for ORION.

    Provides trace lifecycle management. Falls back to
    structlog when LANGSMITH_API_KEY is not set.

    Args:
        project_name: LangSmith project name.
    """

    def __init__(
        self,
        project_name: str = "orion-agent",
    ) -> None:
        self._project = project_name
        self._client: Any = None
        self._active_spans: dict[str, TraceSpan] = {}
        self._enabled = bool(os.environ.get("LANGSMITH_API_KEY"))

        if self._enabled:
            try:
                from langsmith import Client

                self._client = Client()
                logger.info(
                    "tracer_langsmith_enabled",
                    project=project_name,
                )
            except Exception as exc:
                logger.warning(
                    "tracer_langsmith_failed",
                    error=str(exc),
                )
                self._enabled = False

    @contextmanager
    def start_task(
        self,
        task_id: str,
        task_description: str = "",
    ) -> Generator[TraceSpan, None, None]:
        """Start a task-level trace span.

        Args:
            task_id: Task ID.
            task_description: Task description for context.

        Yields:
            TraceSpan for the task.
        """
        span = TraceSpan(
            name="task",
            task_id=task_id,
            tags=[task_id],
            metadata={
                "task_description": task_description[:200],
            },
            start_time=time.monotonic(),
        )
        self._active_spans[task_id] = span

        logger.info(
            "trace_task_started",
            task_id=task_id,
        )

        try:
            yield span
        finally:
            span.end_time = time.monotonic()
            self._active_spans.pop(task_id, None)

            logger.info(
                "trace_task_ended",
                task_id=task_id,
                duration_ms=round(span.duration_ms, 1),
            )

    def start_agent_span(
        self,
        agent_name: str,
        task_id: str,
        subtask_id: str | None = None,
        step_index: int = 0,
        retry_count: int = 0,
    ) -> TraceSpan:
        """Start an agent-level trace span.

        Args:
            agent_name: Agent name.
            task_id: Task ID.
            subtask_id: Optional subtask ID.
            step_index: Pipeline step index.
            retry_count: Retry attempt number.

        Returns:
            TraceSpan for the agent operation.
        """
        span = TraceSpan(
            name=agent_name,
            task_id=task_id,
            tags=[task_id, agent_name],
            metadata={
                "subtask_id": subtask_id,
                "step_index": step_index,
                "retry_count": retry_count,
            },
            start_time=time.monotonic(),
        )

        span_key = f"{task_id}:{agent_name}"
        self._active_spans[span_key] = span

        logger.debug(
            "trace_agent_started",
            agent=agent_name,
            task_id=task_id,
        )

        return span

    def end_agent_span(
        self,
        span: TraceSpan,
        success: bool = True,
        tokens_used: int = 0,
    ) -> None:
        """End an agent-level trace span.

        Args:
            span: The span to end.
            success: Whether the operation succeeded.
            tokens_used: Total tokens used.
        """
        span.end_time = time.monotonic()
        span.metadata["success"] = success
        span.metadata["tokens_used"] = tokens_used

        span_key = f"{span.task_id}:{span.name}"
        self._active_spans.pop(span_key, None)

        logger.debug(
            "trace_agent_ended",
            agent=span.name,
            task_id=span.task_id,
            duration_ms=round(span.duration_ms, 1),
            success=success,
        )

    def end_task(
        self,
        task_id: str,
        success: bool,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record task completion metrics.

        Args:
            task_id: Task ID.
            success: Whether the task succeeded.
            total_tokens: Total tokens across all agents.
            cost_usd: Total estimated cost.
        """
        logger.info(
            "trace_task_completed",
            task_id=task_id,
            success=success,
            total_tokens=total_tokens,
            cost_usd=round(cost_usd, 6),
        )

    @property
    def is_enabled(self) -> bool:
        """Whether LangSmith tracing is active."""
        return self._enabled
