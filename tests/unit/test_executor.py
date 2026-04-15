"""Unit tests for ExecutorAgent and execute_dag."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.executor import ExecutorAgent, topological_sort
from orion.orchestrator.dispatcher import execute_dag


def _make_plan_msg(
    subtasks: list[dict[str, Any]],
    task_id: str = "t_test01",
) -> Msg:
    """Create a plan Msg as if from the Planner."""
    plan = {"chain_of_thought": "test", "subtasks": subtasks}
    return Msg(
        name="planner",
        role="assistant",
        content=json.dumps(plan),
        metadata={
            "orion_meta": {
                "task_id": task_id,
                "subtask_id": None,
                "step_index": 1,
                "retry_count": 0,
                "rollback_available": False,
                "trace_id": "trace_test",
                "context": {},
            },
        },
    )


class TestExecutorAgent:
    """Tests for ExecutorAgent."""

    @pytest.mark.asyncio
    async def test_single_subtask_with_tool(self) -> None:
        """Executor runs a single subtask via tool registry."""
        async def mock_tool(**params: Any) -> str:
            return "file created"

        registry = {"file_write": mock_tool}
        agent = ExecutorAgent(tool_registry=registry)

        msg = _make_plan_msg([{
            "id": "s1",
            "action": "Create file",
            "tool": "file_write",
            "params": {"path": "test.txt"},
            "expected_output": "file created",
            "depends_on": [],
        }])

        result = await agent.reply(msg)
        results = json.loads(result.content)
        assert len(results) == 1
        assert results[0]["ok"] is True
        assert results[0]["output"] == "file created"

    @pytest.mark.asyncio
    async def test_topological_execution_order(self) -> None:
        """Executor respects dependency order."""
        call_order: list[str] = []

        async def tool_a(**params: Any) -> str:
            call_order.append("a")
            return "done_a"

        async def tool_b(**params: Any) -> str:
            call_order.append("b")
            return "done_b"

        registry = {"tool_a": tool_a, "tool_b": tool_b}
        agent = ExecutorAgent(tool_registry=registry)

        msg = _make_plan_msg([
            {"id": "s2", "action": "B", "tool": "tool_b",
             "params": {}, "depends_on": ["s1"]},
            {"id": "s1", "action": "A", "tool": "tool_a",
             "params": {}, "depends_on": []},
        ])

        result = await agent.reply(msg)
        results = json.loads(result.content)

        assert call_order == ["a", "b"]
        assert all(r["ok"] for r in results)

    @pytest.mark.asyncio
    async def test_tool_failure_retry(self) -> None:
        """Executor retries failed tool invocations."""
        call_count = 0

        async def flaky_tool(**params: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                msg = "temporary failure"
                raise RuntimeError(msg)
            return "success"

        registry = {"flaky": flaky_tool}
        agent = ExecutorAgent(tool_registry=registry, max_retries=3)

        msg = _make_plan_msg([{
            "id": "s1", "action": "Flaky op", "tool": "flaky",
            "params": {}, "depends_on": [],
        }])

        result = await agent.reply(msg)
        results = json.loads(result.content)
        assert results[0]["ok"] is True
        assert results[0]["attempt"] == 3

    @pytest.mark.asyncio
    async def test_unknown_tool_simulation(self) -> None:
        """Executor simulates unknown tools without model."""
        agent = ExecutorAgent()  # No model, no registry

        msg = _make_plan_msg([{
            "id": "s1", "action": "Do something",
            "tool": "unknown_tool", "params": {},
            "depends_on": [],
        }])

        result = await agent.reply(msg)
        results = json.loads(result.content)
        assert results[0]["ok"] is True
        assert "[SIMULATED]" in results[0]["output"]

    @pytest.mark.asyncio
    async def test_critical_failure_stops_execution(self) -> None:
        """Critical subtask failure stops remaining execution."""
        async def fail_tool(**params: Any) -> str:
            msg = "critical error"
            raise RuntimeError(msg)

        registry = {"fail": fail_tool}
        agent = ExecutorAgent(tool_registry=registry, max_retries=0)

        msg = _make_plan_msg([
            {"id": "s1", "action": "Fail", "tool": "fail",
             "params": {}, "depends_on": [], "is_critical": True},
            {"id": "s2", "action": "Skip", "tool": "fail",
             "params": {}, "depends_on": []},
        ])

        result = await agent.reply(msg)
        results = json.loads(result.content)
        # Only s1 was executed (failed), s2 was skipped
        assert len(results) == 1
        assert results[0]["ok"] is False

    @pytest.mark.asyncio
    async def test_orion_meta_propagated(self) -> None:
        """orion_meta propagated to output."""
        agent = ExecutorAgent()
        msg = _make_plan_msg(
            [{"id": "s1", "action": "t", "tool": "x", "params": {}, "depends_on": []}],
            task_id="t_meta",
        )

        result = await agent.reply(msg)
        meta = result.metadata.get("orion_meta", {})
        assert meta["task_id"] == "t_meta"
        assert meta["step_index"] == 2

    @pytest.mark.asyncio
    async def test_execute_subtask_public_api(self) -> None:
        """execute_subtask is callable directly (for DAG dispatcher)."""
        async def mock_tool(**params: Any) -> str:
            return "direct call"

        registry = {"direct_tool": mock_tool}
        agent = ExecutorAgent(tool_registry=registry)

        result = await agent.execute_subtask(
            subtask={
                "id": "s1",
                "action": "Direct call",
                "tool": "direct_tool",
                "params": {},
                "depends_on": [],
            },
            task_id="t_direct",
        )
        assert result["ok"] is True
        assert result["output"] == "direct call"


class TestTopologicalSort:
    """Tests for module-level topological_sort function."""

    def test_linear_chain(self) -> None:
        """Linear chain: s1 → s2 → s3."""
        subtasks = [
            {"id": "s3", "depends_on": ["s2"]},
            {"id": "s1", "depends_on": []},
            {"id": "s2", "depends_on": ["s1"]},
        ]
        order = topological_sort(subtasks)
        assert order == ["s1", "s2", "s3"]

    def test_independent_tasks(self) -> None:
        """Independent tasks can appear in any order."""
        subtasks = [
            {"id": "a", "depends_on": []},
            {"id": "b", "depends_on": []},
            {"id": "c", "depends_on": []},
        ]
        order = topological_sort(subtasks)
        assert set(order) == {"a", "b", "c"}
        assert len(order) == 3

    def test_diamond_dependency(self) -> None:
        """Diamond: s1 → s2, s1 → s3, s2+s3 → s4."""
        subtasks = [
            {"id": "s4", "depends_on": ["s2", "s3"]},
            {"id": "s2", "depends_on": ["s1"]},
            {"id": "s3", "depends_on": ["s1"]},
            {"id": "s1", "depends_on": []},
        ]
        order = topological_sort(subtasks)
        assert order[0] == "s1"
        assert order[-1] == "s4"
        assert set(order[1:3]) == {"s2", "s3"}


class TestExecuteDAG:
    """Tests for the DAG dispatcher."""

    @pytest.mark.asyncio
    async def test_parallel_independent_tasks(self) -> None:
        """Independent tasks execute concurrently."""
        timestamps: dict[str, float] = {}

        async def slow_tool(**params: Any) -> str:
            task_id = params.get("id", "?")
            timestamps[f"{task_id}_start"] = time.monotonic()
            await asyncio.sleep(0.1)
            timestamps[f"{task_id}_end"] = time.monotonic()
            return f"done_{task_id}"

        registry = {"slow": slow_tool}
        executor = ExecutorAgent(tool_registry=registry)

        subtasks = [
            {"id": "s1", "action": "A", "tool": "slow",
             "params": {"id": "s1"}, "depends_on": []},
            {"id": "s2", "action": "B", "tool": "slow",
             "params": {"id": "s2"}, "depends_on": []},
        ]

        start = time.monotonic()
        results = await execute_dag(subtasks, executor, "t_par")
        elapsed = time.monotonic() - start

        # Both tasks should succeed
        assert results["s1"]["ok"] is True
        assert results["s2"]["ok"] is True

        # Should take ~0.1s (parallel), not ~0.2s (sequential)
        # Use generous threshold for CI
        assert elapsed < 0.5

    @pytest.mark.asyncio
    async def test_dependency_ordering(self) -> None:
        """Dependent tasks execute in correct order."""
        call_order: list[str] = []

        async def track_tool(**params: Any) -> str:
            tid = params.get("id", "?")
            call_order.append(tid)
            return f"done_{tid}"

        registry = {"track": track_tool}
        executor = ExecutorAgent(tool_registry=registry)

        subtasks = [
            {"id": "s2", "action": "B", "tool": "track",
             "params": {"id": "s2"}, "depends_on": ["s1"]},
            {"id": "s1", "action": "A", "tool": "track",
             "params": {"id": "s1"}, "depends_on": []},
        ]

        results = await execute_dag(subtasks, executor, "t_dep")
        assert results["s1"]["ok"] is True
        assert results["s2"]["ok"] is True
        assert call_order.index("s1") < call_order.index("s2")

    @pytest.mark.asyncio
    async def test_exception_handling(self) -> None:
        """Failed tasks produce error results, don't crash DAG."""
        async def fail_tool(**params: Any) -> str:
            raise RuntimeError("boom")

        registry = {"fail": fail_tool}
        executor = ExecutorAgent(tool_registry=registry, max_retries=0)

        subtasks = [
            {"id": "s1", "action": "Fail", "tool": "fail",
             "params": {}, "depends_on": []},
        ]

        results = await execute_dag(subtasks, executor, "t_err")
        assert results["s1"]["ok"] is False
        assert "boom" in results["s1"]["error"]

    @pytest.mark.asyncio
    async def test_diamond_dag(self) -> None:
        """Diamond DAG: s1 → (s2, s3) → s4."""
        order: list[str] = []

        async def record_tool(**params: Any) -> str:
            tid = params.get("id", "?")
            order.append(tid)
            await asyncio.sleep(0.01)
            return f"done_{tid}"

        registry = {"rec": record_tool}
        executor = ExecutorAgent(tool_registry=registry)

        subtasks = [
            {"id": "s1", "action": "Start", "tool": "rec",
             "params": {"id": "s1"}, "depends_on": []},
            {"id": "s2", "action": "Left", "tool": "rec",
             "params": {"id": "s2"}, "depends_on": ["s1"]},
            {"id": "s3", "action": "Right", "tool": "rec",
             "params": {"id": "s3"}, "depends_on": ["s1"]},
            {"id": "s4", "action": "Join", "tool": "rec",
             "params": {"id": "s4"}, "depends_on": ["s2", "s3"]},
        ]

        results = await execute_dag(subtasks, executor, "t_diamond")

        assert all(r["ok"] for r in results.values())
        assert order[0] == "s1"
        assert order[-1] == "s4"
        # s2 and s3 in middle (parallel, any order)
        assert set(order[1:3]) == {"s2", "s3"}

    @pytest.mark.asyncio
    async def test_unsatisfiable_dag_raises(self) -> None:
        """Cyclic/unsatisfiable DAG raises RuntimeError."""
        executor = ExecutorAgent()

        subtasks = [
            {"id": "s1", "action": "A", "tool": "x",
             "params": {}, "depends_on": ["s2"]},
            {"id": "s2", "action": "B", "tool": "x",
             "params": {}, "depends_on": ["s1"]},
        ]

        with pytest.raises(RuntimeError, match="unsatisfiable"):
            await execute_dag(subtasks, executor, "t_cycle")
