"""Unit tests for ExecutorAgent."""
from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.executor import ExecutorAgent


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
