"""Integration test — full HiClaw pipeline end-to-end.

Tests the complete Planner → Executor → Verifier → Supervisor flow
without real LLM calls (all mocked). Verifies message passing,
orion_meta propagation, and decision flow.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.executor import ExecutorAgent
from orion.agents.planner import PlannerAgent
from orion.agents.supervisor import SupervisorAgent
from orion.agents.verifier import VerifierAgent


class MockPlannerModel:
    """Mock model that returns a valid plan."""

    async def __call__(self, messages: Any = None, **kwargs: Any) -> Any:
        from agentscope.message import TextBlock

        plan = json.dumps({
            "chain_of_thought": "User wants to create a file.",
            "subtasks": [
                {
                    "id": "s1",
                    "action": "Create test.txt",
                    "tool": "file_write",
                    "params": {"path": "test.txt", "content": "hello"},
                    "expected_output": "File created",
                    "depends_on": [],
                    "is_destructive": False,
                },
            ],
        })

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.content = [{"text": text}]

        return FakeResponse(plan)


class TestFullPipeline:
    """Integration tests for the complete HiClaw pipeline."""

    @pytest.mark.asyncio
    async def test_happy_path_pipeline(self) -> None:
        """Full pipeline: plan → execute → verify → supervise."""

        # 1. Planner generates a plan
        planner = PlannerAgent(model=MockPlannerModel())
        initial_msg = Msg(
            name="user",
            role="user",
            content="Create a file called test.txt with 'hello'",
            metadata={
                "orion_meta": {
                    "task_id": "t_integration",
                    "subtask_id": None,
                    "step_index": 0,
                    "retry_count": 0,
                    "rollback_available": False,
                    "trace_id": "trace_int",
                    "context": {},
                },
            },
        )

        plan_msg = await planner.reply(initial_msg)
        plan = json.loads(plan_msg.content)
        assert "subtasks" in plan
        assert len(plan["subtasks"]) == 1

        # Verify meta propagation
        meta = plan_msg.metadata["orion_meta"]
        assert meta["task_id"] == "t_integration"
        assert meta["step_index"] == 1

        # 2. Executor runs the plan (with mock tool)
        async def mock_file_write(**params: Any) -> str:
            return "File created"

        executor = ExecutorAgent(tool_registry={"file_write": mock_file_write})
        exec_msg = await executor.reply(plan_msg)
        results = json.loads(exec_msg.content)
        assert len(results) == 1
        assert results[0]["ok"] is True

        exec_meta = exec_msg.metadata["orion_meta"]
        assert exec_meta["step_index"] == 2

        # 3. Verifier checks the results
        verifier = VerifierAgent()
        verify_msg = await verifier.reply(exec_msg)
        report = json.loads(verify_msg.content)
        assert report["overall"] == "PASS"
        assert report["recommendation"] == "DONE"

        verify_meta = verify_msg.metadata["orion_meta"]
        assert verify_meta["step_index"] == 3

        # 4. Supervisor makes final decision
        supervisor = SupervisorAgent()
        final_msg = await supervisor.reply(verify_msg)
        decision = json.loads(final_msg.content)
        assert decision["decision"]["action"] == "COMPLETE"
        assert decision["decision"]["status"] == "completed"

        final_meta = final_msg.metadata["orion_meta"]
        assert final_meta["step_index"] == 4
        assert final_meta["task_id"] == "t_integration"

    @pytest.mark.asyncio
    async def test_failure_pipeline(self) -> None:
        """Pipeline handles tool failure → HARD_FAIL → ESCALATE."""

        # 1. Planner
        planner = PlannerAgent(model=MockPlannerModel())
        plan_msg = await planner.reply(Msg(
            name="user", role="user",
            content="Create test.txt",
            metadata={"orion_meta": {
                "task_id": "t_fail", "step_index": 0,
                "retry_count": 0, "rollback_available": False,
                "trace_id": "trace_fail", "context": {},
            }},
        ))

        # 2. Executor with failing tool
        async def failing_tool(**params: Any) -> str:
            msg = "disk full"
            raise RuntimeError(msg)

        executor = ExecutorAgent(
            tool_registry={"file_write": failing_tool},
            max_retries=0,
        )
        exec_msg = await executor.reply(plan_msg)
        results = json.loads(exec_msg.content)
        assert results[0]["ok"] is False

        # 3. Verifier detects failure
        verifier = VerifierAgent()
        verify_msg = await verifier.reply(exec_msg)
        report = json.loads(verify_msg.content)
        assert report["overall"] == "HARD_FAIL"

        # 4. Supervisor escalates (no rollback available)
        supervisor = SupervisorAgent()
        final_msg = await supervisor.reply(verify_msg)
        decision = json.loads(final_msg.content)
        assert decision["decision"]["action"] == "ESCALATE"

    @pytest.mark.asyncio
    async def test_meta_chain_integrity(self) -> None:
        """task_id propagates across all 4 agents."""
        planner = PlannerAgent(model=MockPlannerModel())

        msg = Msg(
            name="user", role="user", content="test",
            metadata={"orion_meta": {
                "task_id": "t_chain_check",
                "step_index": 0, "retry_count": 0,
                "rollback_available": False,
                "trace_id": "trace_chain", "context": {},
            }},
        )

        plan_msg = await planner.reply(msg)
        executor = ExecutorAgent()
        exec_msg = await executor.reply(plan_msg)
        verifier = VerifierAgent()
        verify_msg = await verifier.reply(exec_msg)
        supervisor = SupervisorAgent()
        final_msg = await supervisor.reply(verify_msg)

        # All agents should preserve the same task_id
        for m in [plan_msg, exec_msg, verify_msg, final_msg]:
            assert m.metadata["orion_meta"]["task_id"] == "t_chain_check"

        # Step index should increment: 1, 2, 3, 4
        assert plan_msg.metadata["orion_meta"]["step_index"] == 1
        assert exec_msg.metadata["orion_meta"]["step_index"] == 2
        assert verify_msg.metadata["orion_meta"]["step_index"] == 3
        assert final_msg.metadata["orion_meta"]["step_index"] == 4
