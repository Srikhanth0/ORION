"""Unit tests for PlannerAgent."""
from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.planner import PlannerAgent
from orion.core.exceptions import PlanError


def _make_msg(
    content: str = "Create a file called test.txt",
    task_id: str = "t_test01",
) -> Msg:
    """Create a test input Msg."""
    return Msg(
        name="user",
        role="user",
        content=content,
        metadata={
            "orion_meta": {
                "task_id": task_id,
                "subtask_id": None,
                "step_index": 0,
                "retry_count": 0,
                "rollback_available": False,
                "trace_id": "trace_test",
                "context": {},
            },
        },
    )


class MockModel:
    """Mock model that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def __call__(self, messages: Any = None, **kwargs: Any) -> Any:
        from agentscope.message import TextBlock

        idx = min(self._call_count, len(self._responses) - 1)
        self._call_count += 1

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.content = [{"text": text}]
        return FakeResponse(self._responses[idx])


class TestPlannerAgent:
    """Tests for PlannerAgent."""

    @pytest.mark.asyncio
    async def test_valid_plan_generation(self) -> None:
        """Planner produces valid structured plan."""
        valid_plan = json.dumps({
            "chain_of_thought": "Need to create a file",
            "subtasks": [
                {
                    "id": "s1",
                    "action": "Create file test.txt",
                    "tool": "file_write",
                    "params": {"path": "test.txt", "content": "hello"},
                    "expected_output": "File created",
                    "depends_on": [],
                    "is_destructive": False,
                }
            ],
        })

        model = MockModel([valid_plan])
        agent = PlannerAgent(model=model)
        result = await agent.reply(_make_msg())

        plan = json.loads(result.content)
        assert "subtasks" in plan
        assert len(plan["subtasks"]) == 1
        assert plan["subtasks"][0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_plan_with_dependencies(self) -> None:
        """Planner handles multi-step plans with dependencies."""
        plan = json.dumps({
            "chain_of_thought": "First create dir, then file",
            "subtasks": [
                {
                    "id": "s1",
                    "action": "Create directory",
                    "tool": "mkdir",
                    "params": {"path": "mydir"},
                    "expected_output": "Directory created",
                    "depends_on": [],
                    "is_destructive": False,
                },
                {
                    "id": "s2",
                    "action": "Create file in directory",
                    "tool": "file_write",
                    "params": {"path": "mydir/test.txt"},
                    "expected_output": "File created",
                    "depends_on": ["s1"],
                    "is_destructive": False,
                },
            ],
        })

        model = MockModel([plan])
        agent = PlannerAgent(model=model)
        result = await agent.reply(_make_msg())

        parsed = json.loads(result.content)
        assert len(parsed["subtasks"]) == 2
        assert parsed["subtasks"][1]["depends_on"] == ["s1"]

    @pytest.mark.asyncio
    async def test_plan_validation_missing_subtasks(self) -> None:
        """Planner rejects plan without subtasks."""
        model = MockModel(['{"chain_of_thought": "dunno"}'] * 3)
        agent = PlannerAgent(model=model, max_retries=1)

        with pytest.raises(PlanError, match="subtasks"):
            await agent.reply(_make_msg())

    @pytest.mark.asyncio
    async def test_plan_validation_duplicate_ids(self) -> None:
        """Planner rejects plan with duplicate subtask IDs."""
        bad_plan = json.dumps({
            "subtasks": [
                {"id": "s1", "action": "a", "tool": "t"},
                {"id": "s1", "action": "b", "tool": "t"},
            ],
        })
        model = MockModel([bad_plan] * 3)
        agent = PlannerAgent(model=model, max_retries=1)

        with pytest.raises(PlanError, match="Duplicate"):
            await agent.reply(_make_msg())

    @pytest.mark.asyncio
    async def test_orion_meta_propagated(self) -> None:
        """orion_meta is propagated to output message."""
        plan = json.dumps({
            "subtasks": [{"id": "s1", "action": "test", "tool": "t"}],
        })
        model = MockModel([plan])
        agent = PlannerAgent(model=model)
        result = await agent.reply(_make_msg(task_id="t_abc"))

        meta = result.metadata.get("orion_meta", {})
        assert meta["task_id"] == "t_abc"
        assert meta["step_index"] == 1

    @pytest.mark.asyncio
    async def test_retry_on_parse_error(self) -> None:
        """Planner retries on JSON parse failure."""
        valid_plan = json.dumps({
            "subtasks": [{"id": "s1", "action": "test", "tool": "t"}],
        })
        model = MockModel(["not json!!!", valid_plan])
        agent = PlannerAgent(model=model, max_retries=1)
        result = await agent.reply(_make_msg())

        parsed = json.loads(result.content)
        assert len(parsed["subtasks"]) == 1
