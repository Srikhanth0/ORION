"""Shared test fixtures for the ORION test suite.

Provides mock objects for the LLM router, tool registry, and
reusable Task/Subtask/TaskDAG instances for unit tests.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from orion.core.result import RollbackPoint, StepResult, TaskResult
from orion.core.task import Subtask, Task, TaskDAG, TaskStatus

# ── Mock LLM Router ─────────────────────────────────────────


class MockLLMRouter:
    """A fake LLM router that returns pre-baked responses.

    Tracks call count and the messages it received for assertions.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [
            '{"subtasks": [{"id": "s1", "action": "test action", "tool_hint": "echo"}]}'
        ]
        self.call_count = 0
        self.call_log: list[list[dict[str, Any]]] = []

    async def chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str:
        """Return the next pre-baked response.

        Args:
            messages: The chat messages (recorded for assertions).
            **kwargs: Additional arguments (ignored).

        Returns:
            The next response from the responses list (cycles).
        """
        self.call_log.append(messages)
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return response

    async def is_healthy(self) -> bool:
        """Always healthy in tests."""
        return True


# ── Mock Tool Registry ───────────────────────────────────────


class MockToolRegistry:
    """A fake tool registry backed by a simple dict.

    Each tool is a callable that receives params and returns a dict.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {
            "echo": AsyncMock(return_value={"output": "hello world"}),
            "create_file": AsyncMock(return_value={"path": "/tmp/test.txt"}),
            "list_files": AsyncMock(return_value={"files": ["a.txt", "b.txt"]}),
        }

    def get(self, name: str) -> Any:
        """Get a mock tool by name.

        Args:
            name: Tool name to look up.

        Returns:
            The mock callable.

        Raises:
            KeyError: If tool not found.
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not in mock registry")
        return self._tools[name]

    def register(self, name: str, tool: Any) -> None:
        """Register a mock tool.

        Args:
            name: Tool name.
            tool: Mock callable.
        """
        self._tools[name] = tool


# ── Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def mock_llm_router() -> MockLLMRouter:
    """Provide a mock LLM router with default responses."""
    return MockLLMRouter()


@pytest.fixture()
def mock_tool_registry() -> MockToolRegistry:
    """Provide a mock tool registry with echo/create_file/list_files."""
    return MockToolRegistry()


@pytest.fixture()
def sample_subtask_a() -> Subtask:
    """A simple subtask with no dependencies."""
    return Subtask(id="s1", action="Clone repository", tool_hint="github_clone")


@pytest.fixture()
def sample_subtask_b() -> Subtask:
    """A subtask that depends on s1."""
    return Subtask(
        id="s2",
        action="Run tests",
        tool_hint="exec_cmd",
        depends_on=["s1"],
        params={"command": "pytest"},
    )


@pytest.fixture()
def sample_subtask_c() -> Subtask:
    """A subtask that depends on s2."""
    return Subtask(
        id="s3",
        action="Deploy artifact",
        tool_hint="push_files",
        depends_on=["s2"],
    )


@pytest.fixture()
def sample_dag(
    sample_subtask_a: Subtask,
    sample_subtask_b: Subtask,
    sample_subtask_c: Subtask,
) -> TaskDAG:
    """A 3-node linear DAG: s1 → s2 → s3."""
    return TaskDAG(
        task_id="t_test001",
        instruction="Clone repo, run tests, deploy",
        subtasks=[sample_subtask_a, sample_subtask_b, sample_subtask_c],
    )


@pytest.fixture()
def sample_task(sample_dag: TaskDAG) -> Task:
    """A Task with a pre-built DAG in PLANNING status."""
    return Task(
        task_id="t_test001",
        instruction="Clone repo, run tests, deploy",
        dag=sample_dag,
        status=TaskStatus.PLANNING,
    )


@pytest.fixture()
def sample_step_result_success() -> StepResult:
    """A successful StepResult."""
    return StepResult(
        subtask_id="s1",
        ok=True,
        output={"cloned": True, "path": "/workspace/repo"},
        duration_ms=1200,
        provider_used="groq",
    )


@pytest.fixture()
def sample_step_result_failure() -> StepResult:
    """A failed StepResult."""
    return StepResult(
        subtask_id="s2",
        ok=False,
        error="pytest returned exit code 1",
        duration_ms=5400,
        provider_used="openrouter",
        attempt=2,
    )


@pytest.fixture()
def sample_rollback_point() -> RollbackPoint:
    """A file-based rollback checkpoint."""
    return RollbackPoint(
        subtask_id="s1",
        checkpoint_type="file",
        checkpoint_data={"path": "/workspace/repo", "hash": "abc123"},
    )


@pytest.fixture()
def sample_task_result(
    sample_step_result_success: StepResult,
    sample_step_result_failure: StepResult,
) -> TaskResult:
    """An aggregate TaskResult with mixed step outcomes."""
    return TaskResult(
        task_id="t_test001",
        status=TaskStatus.FAILED,
        step_results=[sample_step_result_success, sample_step_result_failure],
        total_duration_ms=6600,
        total_cost_usd=0.003,
        verification_passed=False,
        verification_notes="Step s2 failed: test suite did not pass.",
    )
