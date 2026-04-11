"""Unit tests for orion.core.exceptions — exception hierarchy.

Tests cover:
- Structured context on every exception type
- Inheritance hierarchy correctness
- Domain-specific fields (cycle_path, tool_name, provider, etc.)
- Exception message formatting
"""
from __future__ import annotations

import pytest

from orion.core.exceptions import (
    AllProvidersExhaustedError,
    LLMError,
    OrionError,
    PermissionDeniedError,
    PlanError,
    QuotaExceededError,
    RollbackError,
    SafetyError,
    SandboxViolationError,
    TaskValidationError,
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
)


class TestOrionError:
    """Tests for the base OrionError."""

    def test_base_error_message(self) -> None:
        """OrionError stores message and structured context."""
        e = OrionError("something broke", task_id="t1", subtask_id="s1")
        assert str(e) == "something broke"
        assert e.task_id == "t1"
        assert e.subtask_id == "s1"
        assert e.context == {}

    def test_base_error_with_context(self) -> None:
        """OrionError accepts arbitrary context dict."""
        ctx = {"provider": "groq", "attempt": 3}
        e = OrionError("fail", context=ctx)
        assert e.context == ctx

    def test_base_error_defaults(self) -> None:
        """OrionError defaults for optional fields."""
        e = OrionError("fail")
        assert e.task_id is None
        assert e.subtask_id is None
        assert e.context == {}


class TestHierarchy:
    """Tests that the exception hierarchy is correct."""

    @pytest.mark.parametrize(
        ("exc_class", "parent_class"),
        [
            (TaskValidationError, OrionError),
            (PlanError, OrionError),
            (ToolError, OrionError),
            (ToolNotFoundError, ToolError),
            (ToolTimeoutError, ToolError),
            (SafetyError, OrionError),
            (PermissionDeniedError, SafetyError),
            (SandboxViolationError, SafetyError),
            (LLMError, OrionError),
            (AllProvidersExhaustedError, LLMError),
            (QuotaExceededError, LLMError),
            (RollbackError, OrionError),
        ],
    )
    def test_inheritance(
        self, exc_class: type[OrionError], parent_class: type[Exception]
    ) -> None:
        """Each exception is a subclass of its documented parent."""
        assert issubclass(exc_class, parent_class)
        assert issubclass(exc_class, OrionError)
        assert issubclass(exc_class, Exception)


class TestTaskValidationError:
    """Tests for TaskValidationError domain-specific fields."""

    def test_cycle_path(self) -> None:
        """TaskValidationError stores cycle_path."""
        e = TaskValidationError(
            "cycle detected",
            task_id="t1",
            cycle_path=["s1", "s2", "s3"],
        )
        assert e.cycle_path == ["s1", "s2", "s3"]

    def test_cycle_path_default(self) -> None:
        """cycle_path defaults to None."""
        e = TaskValidationError("no cycle")
        assert e.cycle_path is None


class TestToolErrors:
    """Tests for tool error domain-specific fields."""

    def test_tool_error_name(self) -> None:
        """ToolError stores the tool name."""
        e = ToolError("invoke failed", tool_name="github_clone")
        assert e.tool_name == "github_clone"

    def test_tool_not_found_available_tools(self) -> None:
        """ToolNotFoundError stores available tool names."""
        e = ToolNotFoundError(
            "not found",
            tool_name="nonexistent",
            available_tools=["echo", "create_file"],
        )
        assert e.tool_name == "nonexistent"
        assert "echo" in e.available_tools

    def test_tool_timeout_durations(self) -> None:
        """ToolTimeoutError stores timeout and elapsed durations."""
        e = ToolTimeoutError(
            "timed out",
            tool_name="slow_tool",
            timeout_seconds=30.0,
            elapsed_seconds=31.5,
        )
        assert e.timeout_seconds == 30.0
        assert e.elapsed_seconds == 31.5


class TestSafetyErrors:
    """Tests for safety error domain-specific fields."""

    def test_permission_denied(self) -> None:
        """PermissionDeniedError stores tool and action."""
        e = PermissionDeniedError(
            "blocked",
            tool_name="delete_repo",
            action="delete",
            rule="github.denied",
        )
        assert e.tool_name == "delete_repo"
        assert e.action == "delete"
        assert e.rule == "github.denied"

    def test_sandbox_violation(self) -> None:
        """SandboxViolationError stores resource, limit, actual."""
        e = SandboxViolationError(
            "memory exceeded",
            resource="memory",
            limit="512MB",
            actual="768MB",
        )
        assert e.resource == "memory"
        assert e.limit == "512MB"
        assert e.actual == "768MB"


class TestLLMErrors:
    """Tests for LLM error domain-specific fields."""

    def test_all_providers_exhausted(self) -> None:
        """AllProvidersExhaustedError stores attempted providers and errors."""
        e = AllProvidersExhaustedError(
            "all failed",
            attempted_providers=["vllm", "groq", "openrouter"],
            errors={"vllm": "offline", "groq": "429", "openrouter": "timeout"},
        )
        assert len(e.attempted_providers) == 3
        assert e.errors["groq"] == "429"

    def test_quota_exceeded(self) -> None:
        """QuotaExceededError stores remaining quota and retry-after."""
        e = QuotaExceededError(
            "rate limited",
            provider="groq",
            remaining_requests=0,
            remaining_tokens=500,
            retry_after_seconds=30.0,
        )
        assert e.provider == "groq"
        assert e.remaining_requests == 0
        assert e.retry_after_seconds == 30.0


class TestRollbackError:
    """Tests for RollbackError domain-specific fields."""

    def test_checkpoint_path(self) -> None:
        """RollbackError stores the checkpoint path."""
        e = RollbackError(
            "restore failed",
            checkpoint_path="/tmp/checkpoints/s1.json",
        )
        assert e.checkpoint_path == "/tmp/checkpoints/s1.json"
