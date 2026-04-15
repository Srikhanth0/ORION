"""Unit tests for Phase 5 — circuit breaker, SSE events, MCP retry,
rollback, logging, and metrics.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from orion.api.schemas import EventType
from orion.llm.router import CircuitBreaker, CircuitBreakerState
from orion.observability.logger import TaskContextFilter, configure_logging
from orion.observability.metrics import (
    record_llm_call,
    record_task,
    record_tool_call,
    record_vision_call,
)
from orion.safety.rollback import RollbackEngine

# ── Circuit Breaker Tests ────────────────────────────────


class TestCircuitBreaker:
    """Test 3-state CLOSED → OPEN → HALF_OPEN circuit breaker."""

    @pytest.mark.asyncio
    async def test_starts_closed(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        assert cb.state == CircuitBreakerState.CLOSED
        assert not cb.is_open

    @pytest.mark.asyncio
    async def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED
        tripped = await cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            await cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        await asyncio.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert not cb.is_open  # HALF_OPEN allows trial

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            await cb.record_failure()
        await asyncio.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        await cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
        for _ in range(3):
            await cb.record_failure()
        await asyncio.sleep(0.15)
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Set internal state to HALF_OPEN explicitly (timer-based)
        cb._state = CircuitBreakerState.HALF_OPEN
        tripped = await cb.record_failure()
        assert tripped is True
        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_count(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()
        # Should be back to 0, so 2 more failures shouldn't trip
        await cb.record_failure()
        await cb.record_failure()
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        for _ in range(3):
            await cb.record_failure()
        assert cb.is_open
        await cb.reset()
        assert cb.state == CircuitBreakerState.CLOSED


# ── SSE Event Types ──────────────────────────────────────


class TestSSEEvents:
    """Test expanded SSE event type enum."""

    def test_fine_grained_events_exist(self) -> None:
        assert EventType.PLANNER_START == "planner_start"
        assert EventType.SUBTASK_QUEUED == "subtask_queued"
        assert EventType.SUBTASK_START == "subtask_start"
        assert EventType.SUBTASK_RESULT == "subtask_result"
        assert EventType.VERIFIER_RESULT == "verifier_result"
        assert EventType.SUPERVISOR_DECISION == "supervisor_decision"
        assert EventType.DONE == "done"

    def test_legacy_events_preserved(self) -> None:
        assert EventType.STEP_START == "STEP_START"
        assert EventType.STEP_DONE == "STEP_DONE"
        assert EventType.TASK_DONE == "TASK_DONE"


# ── Rollback Engine ──────────────────────────────────────


class TestRollbackEngine:
    """Test rollback engine with 50-entry max."""

    def test_max_checkpoints_is_50(self) -> None:
        engine = RollbackEngine()
        assert engine._max_checkpoints == 50

    def test_checkpoint_and_rollback(self, tmp_path: Any) -> None:
        engine = RollbackEngine(checkpoint_dir=tmp_path / "ckpt")
        test_file = tmp_path / "test.txt"

        # Checkpoint with no existing file
        engine.checkpoint("s1", "write_file", {"path": str(test_file)}, task_id="t1")

        # Simulate file creation
        test_file.write_text("hello world")
        assert test_file.exists()

        # Rollback should delete the file (it didn't exist before)
        results = engine.rollback("t1")
        assert len(results) == 1
        assert "[OK]" in results[0]
        assert not test_file.exists()

    def test_checkpoint_overflow(self, tmp_path: Any) -> None:
        engine = RollbackEngine(checkpoint_dir=tmp_path / "ckpt", max_checkpoints=3)
        for i in range(5):
            engine.checkpoint(f"s{i}", "write_file", {"path": f"/tmp/f{i}"}, task_id="t1")
        # Only 3 should remain
        assert len(engine._stacks["t1"]) == 3


# ── Logger Context ───────────────────────────────────────


class TestLoggerContext:
    """Test TaskContextFilter with subtask_id."""

    def test_set_context_with_subtask(self) -> None:
        TaskContextFilter.set_context(task_id="t1", agent="Planner", subtask_id="s1")
        assert TaskContextFilter._context["subtask_id"] == "s1"
        assert TaskContextFilter._context["agent"] == "Planner"
        TaskContextFilter.clear_context()

    def test_clear_context(self) -> None:
        TaskContextFilter.set_context(task_id="t1", agent="Executor")
        TaskContextFilter.clear_context()
        assert TaskContextFilter._context == {}

    def test_configure_logging_runs(self) -> None:
        # Smoke test — just ensure it doesn't crash
        configure_logging(level="DEBUG")
        configure_logging(level="INFO", force_json=True)


# ── MCP Client Retry ────────────────────────────────────


class TestMCPRetry:
    """Test MCP client retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self) -> None:
        from orion.tools.mcp_client import MCPClient

        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(
            category=MagicMock(value="os"),
            is_destructive=False,
            params_schema={},
        )

        call_count = 0

        async def flaky_call(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("MCP down")
            return "success"

        mock_registry.call = flaky_call

        client = MCPClient(registry=mock_registry)
        result = await client.invoke("test_tool", {})
        assert result.ok is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self) -> None:
        from orion.tools.mcp_client import MCPClient

        mock_registry = MagicMock()
        mock_registry.get.return_value = MagicMock(
            category=MagicMock(value="os"),
            is_destructive=False,
            params_schema={},
        )

        async def always_fail(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError("Permanent failure")

        mock_registry.call = always_fail

        client = MCPClient(registry=mock_registry)
        result = await client.invoke("test_tool", {})
        assert result.ok is False


# ── Metrics ──────────────────────────────────────────────


class TestMetrics:
    """Test renamed orion_* metrics helpers."""

    def test_record_llm_call_no_crash(self) -> None:
        # Should not crash even if prometheus_client unavailable
        record_llm_call(
            provider="groq", model="llama3", status="success",
            latency_seconds=1.5, input_tokens=100, output_tokens=50,
        )

    def test_record_tool_call_no_crash(self) -> None:
        record_tool_call(tool="exec_cmd", category="os", status="success", latency_seconds=0.3)

    def test_record_task_no_crash(self) -> None:
        record_task(status="pass", duration_seconds=5.0)

    def test_record_vision_call_no_crash(self) -> None:
        record_vision_call(latency_seconds=2.5)


# ── Eval Suite ───────────────────────────────────────────


class TestEvalSuite:
    """Test eval fixture completeness."""

    def test_25_tasks_loaded(self) -> None:
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tasks.json"
        tasks = json.loads(fixtures.read_text())
        assert len(tasks) == 25

    def test_all_categories_covered(self) -> None:
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tasks.json"
        tasks = json.loads(fixtures.read_text())
        categories = {t["category"] for t in tasks}
        assert "os" in categories
        assert "browser" in categories
        assert "github" in categories
        assert "vision" in categories
        assert "multi-step" in categories
        assert "error" in categories

    def test_error_tasks_expect_failure(self) -> None:
        from pathlib import Path

        fixtures = Path(__file__).resolve().parent.parent / "fixtures" / "sample_tasks.json"
        tasks = json.loads(fixtures.read_text())
        error_tasks = [t for t in tasks if t["category"] == "error"]
        assert len(error_tasks) == 3
        assert all(t["expected_status"] == "FAILED" for t in error_tasks)
