"""Unit tests for VerifierAgent."""
from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import Msg

from orion.agents.verifier import VerifierAgent


def _make_results_msg(
    results: list[dict[str, Any]],
    task_id: str = "t_test01",
) -> Msg:
    """Create a results Msg as if from the Executor."""
    return Msg(
        name="executor",
        role="assistant",
        content=json.dumps(results),
        metadata={
            "orion_meta": {
                "task_id": task_id,
                "subtask_id": None,
                "step_index": 2,
                "retry_count": 0,
                "rollback_available": False,
                "trace_id": "trace_test",
                "context": {},
            },
        },
    )


class TestVerifierAgent:
    """Tests for VerifierAgent."""

    @pytest.mark.asyncio
    async def test_all_pass(self) -> None:
        """All steps successful → PASS."""
        agent = VerifierAgent()  # No model — pure assertion mode
        msg = _make_results_msg([
            {"subtask_id": "s1", "ok": True, "output": "file created",
             "expected_output": "file created"},
        ])

        result = await agent.reply(msg)
        report = json.loads(result.content)
        assert report["overall"] == "PASS"
        assert report["recommendation"] == "DONE"

    @pytest.mark.asyncio
    async def test_execution_failure_hard_fail(self) -> None:
        """Execution failure → HARD_FAIL."""
        agent = VerifierAgent()
        msg = _make_results_msg([
            {"subtask_id": "s1", "ok": False, "output": None,
             "error": "tool crashed"},
        ])

        result = await agent.reply(msg)
        report = json.loads(result.content)
        assert report["overall"] == "HARD_FAIL"
        assert report["recommendation"] == "ROLLBACK"

    @pytest.mark.asyncio
    async def test_assertion_mismatch_soft_fail(self) -> None:
        """Output doesn't match expected → SOFT_FAIL."""
        agent = VerifierAgent()
        msg = _make_results_msg([
            {"subtask_id": "s1", "ok": True, "output": "wrong output",
             "expected_output": "expected something else entirely"},
        ])

        result = await agent.reply(msg)
        report = json.loads(result.content)
        assert report["overall"] == "SOFT_FAIL"

    @pytest.mark.asyncio
    async def test_empty_results(self) -> None:
        """Empty results list → PASS (nothing to fail)."""
        agent = VerifierAgent()
        msg = _make_results_msg([])

        result = await agent.reply(msg)
        report = json.loads(result.content)
        assert report["overall"] == "PASS"

    @pytest.mark.asyncio
    async def test_multiple_steps_mixed(self) -> None:
        """Mix of pass and fail steps."""
        agent = VerifierAgent()
        msg = _make_results_msg([
            {"subtask_id": "s1", "ok": True, "output": "done",
             "expected_output": "done"},
            {"subtask_id": "s2", "ok": False, "output": None,
             "error": "failed"},
        ])

        result = await agent.reply(msg)
        report = json.loads(result.content)
        assert report["overall"] == "HARD_FAIL"
        assert len(report["issues"]) > 0

    @pytest.mark.asyncio
    async def test_orion_meta_propagated(self) -> None:
        """orion_meta propagated to output."""
        agent = VerifierAgent()
        msg = _make_results_msg(
            [{"subtask_id": "s1", "ok": True, "output": "ok"}],
            task_id="t_verify",
        )

        result = await agent.reply(msg)
        meta = result.metadata.get("orion_meta", {})
        assert meta["task_id"] == "t_verify"
        assert meta["step_index"] == 3
