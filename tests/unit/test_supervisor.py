"""Unit tests for SupervisorAgent."""

from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.supervisor import SupervisorAgent


def _make_report_msg(
    overall: str = "PASS",
    recommendation: str = "DONE",
    issues: list[dict[str, Any]] | None = None,
    task_id: str = "t_test01",
    retry_count: int = 0,
    rollback_available: bool = False,
) -> Msg:
    """Create a verification report Msg."""
    report = {
        "overall": overall,
        "recommendation": recommendation,
        "issues": issues or [],
    }
    return Msg(
        name="verifier",
        role="assistant",
        content=json.dumps(report),
        metadata={
            "orion_meta": {
                "task_id": task_id,
                "subtask_id": None,
                "step_index": 3,
                "retry_count": retry_count,
                "rollback_available": rollback_available,
                "trace_id": "trace_test",
                "context": {},
            },
        },
    )


class TestSupervisorAgent:
    """Tests for SupervisorAgent."""

    @pytest.mark.asyncio
    async def test_pass_completes_task(self) -> None:
        """PASS → COMPLETE decision."""
        agent = SupervisorAgent()
        result = await agent.reply(_make_report_msg("PASS", "DONE"))

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "COMPLETE"
        assert decision["decision"]["status"] == "completed"  # type: ignore

    @pytest.mark.asyncio
    async def test_soft_fail_retries(self) -> None:
        """SOFT_FAIL with retries left → RETRY decision."""
        agent = SupervisorAgent()
        result = await agent.reply(_make_report_msg("SOFT_FAIL", "RETRY_STEP", retry_count=0))

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "RETRY"

    @pytest.mark.asyncio
    async def test_soft_fail_max_retries_escalates(self) -> None:
        """SOFT_FAIL at max retries → ESCALATE."""
        agent = SupervisorAgent(max_auto_retries=3)
        result = await agent.reply(_make_report_msg("SOFT_FAIL", "RETRY_STEP", retry_count=3))

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "ESCALATE"

    @pytest.mark.asyncio
    async def test_hard_fail_with_rollback(self) -> None:
        """HARD_FAIL with rollback available → ROLLBACK."""
        agent = SupervisorAgent()
        result = await agent.reply(
            _make_report_msg(
                "HARD_FAIL",
                "ROLLBACK",
                rollback_available=True,
            )
        )

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "ROLLBACK"

    @pytest.mark.asyncio
    async def test_hard_fail_without_rollback_escalates(self) -> None:
        """HARD_FAIL without rollback → ESCALATE."""
        agent = SupervisorAgent()
        result = await agent.reply(
            _make_report_msg(
                "HARD_FAIL",
                "ROLLBACK",
                rollback_available=False,
            )
        )

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "ESCALATE"

    @pytest.mark.asyncio
    async def test_unknown_status_escalates(self) -> None:
        """Unknown verification status → ESCALATE."""
        agent = SupervisorAgent()
        result = await agent.reply(_make_report_msg("UNKNOWN_STATUS", "UNKNOWN"))

        decision = json.loads(str(result.content))
        assert decision["decision"]["action"] == "ESCALATE"

    @pytest.mark.asyncio
    async def test_orion_meta_propagated(self) -> None:
        """orion_meta propagated to output."""
        agent = SupervisorAgent()
        result = await agent.reply(_make_report_msg(task_id="t_sup"))

        meta = result.metadata.get("orion_meta", {})
        assert meta["task_id"] == "t_sup"
        assert meta["step_index"] == 4

    @pytest.mark.asyncio
    async def test_issues_forwarded_in_decision(self) -> None:
        """Issues from verifier are included in decision."""
        issues = [{"subtask_id": "s1", "check_type": "exec", "detail": "crashed"}]
        agent = SupervisorAgent()
        result = await agent.reply(
            _make_report_msg(
                "HARD_FAIL",
                "ESCALATE",
                issues=issues,
            )
        )

        decision = json.loads(str(result.content))
        assert len(decision["decision"]["issues"]) == 1
