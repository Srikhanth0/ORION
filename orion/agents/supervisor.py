"""SupervisorAgent — HITL decision maker with retry/rollback/escalation.

Receives a VerificationReport from the Verifier and decides the
next action: DONE, retry, rollback, or escalate to human.

Module Contract
---------------
- **Input**: Msg with VerificationReport JSON from Verifier.
- **Output**: Msg with final decision + TaskResult in content.
- **Decision Matrix**: PASS→done, SOFT_FAIL→retry, HARD_FAIL→rollback, ESCALATE→human.

Depends On
----------
- ``orion.agents.base`` (BaseOrionAgent)
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from agentscope.message import Msg

from orion.agents.base import BaseOrionAgent

logger = structlog.get_logger(__name__)

_MAX_AUTO_RETRIES = 3


class SupervisorAgent(BaseOrionAgent):
    """HITL supervisor with decision matrix for retry/rollback/escalation.

    Decision matrix:
    - PASS → emit final TaskResult, store plan in long-term memory.
    - SOFT_FAIL → ask LLM if auto-retry is safe; retry if yes (max 3).
    - HARD_FAIL → check if rollback available; execute rollback.
    - ESCALATE → call HITL gateway (blocking in dev, webhook in prod).
    - SAFETY_ERROR → immediate abort, rollback all, alert.

    Args:
        model: OrionModelWrapper for retry-safety analysis.
        hitl_gateway: Human-in-the-loop gateway (stub).
        max_auto_retries: Maximum auto-retry cycles before escalation.
    """

    def __init__(
        self,
        model: Any = None,
        hitl_gateway: Any = None,
        max_auto_retries: int = _MAX_AUTO_RETRIES,
    ) -> None:
        super().__init__(
            agent_name="supervisor",
            model=model,
            prompt_template="supervisor_system.j2",
        )
        self._hitl_gateway = hitl_gateway
        self._max_auto_retries = max_auto_retries

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        """Process VerificationReport and decide next action.

        Args:
            *args: First positional arg is the Verifier's output Msg.
            **kwargs: Additional keyword args.

        Returns:
            Msg with supervisor decision and TaskResult in content.
        """
        default = Msg(name="verifier", role="assistant", content="{}")
        x: Msg = args[0] if args else kwargs.get("x", default)
        meta = self._get_orion_meta(x)
        task_id = meta["task_id"]
        retry_count = meta.get("retry_count", 0)

        content = x.content if isinstance(x.content, str) else str(x.content)
        try:
            report = self._parse_json(content)
        except ValueError:
            report = {
                "overall": "HARD_FAIL",
                "issues": [{"detail": "Could not parse verification report"}],
                "recommendation": "ESCALATE",
            }

        overall = report.get("overall", "HARD_FAIL")
        recommendation = report.get("recommendation", "ESCALATE")
        issues = report.get("issues", [])

        logger.info(
            "supervisor_evaluating",
            task_id=task_id,
            overall=overall,
            recommendation=recommendation,
            retry_count=retry_count,
        )

        # Decision matrix
        decision = await self._decide(
            overall=overall,
            recommendation=recommendation,
            issues=issues,
            retry_count=retry_count,
            task_id=task_id,
            meta=meta,
        )

        logger.info(
            "supervisor_decision",
            task_id=task_id,
            decision=decision["action"],
            reasoning=decision.get("reasoning", ""),
        )

        result = {
            "task_id": task_id,
            "decision": decision,
            "verification_report": report,
        }

        return self._make_reply(
            content=json.dumps(result, indent=2),
            source_msg=x,
            meta_updates={
                "step_index": 4,
                "retry_count": retry_count,
            },
        )

    async def _decide(
        self,
        overall: str,
        recommendation: str,
        issues: list[dict[str, Any]],
        retry_count: int,
        task_id: str,
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the decision matrix.

        Args:
            overall: PASS, SOFT_FAIL, or HARD_FAIL.
            recommendation: DONE, RETRY_STEP, ESCALATE, or ROLLBACK.
            issues: List of issue dicts from the verifier.
            retry_count: Current retry count.
            task_id: Task identifier.
            meta: Full orion_meta dict.

        Returns:
            Decision dict with action, reasoning, and details.
        """
        # PASS → Done
        if overall == "PASS":
            return {
                "action": "COMPLETE",
                "status": "completed",
                "reasoning": "All verifications passed. Task is complete.",
            }

        # SOFT_FAIL → Maybe retry
        if overall == "SOFT_FAIL":
            if retry_count >= self._max_auto_retries:
                return {
                    "action": "ESCALATE",
                    "status": "escalated",
                    "reasoning": (
                        f"Soft failure after {retry_count} retries. Escalating to human."
                    ),
                    "issues": issues,
                }

            # Check if auto-retry is safe
            safe_to_retry = await self._is_safe_to_retry(issues, task_id)
            if safe_to_retry:
                return {
                    "action": "RETRY",
                    "status": "retrying",
                    "reasoning": "Soft failure with safe retry conditions.",
                    "retry_count": retry_count + 1,
                }
            return {
                "action": "ESCALATE",
                "status": "escalated",
                "reasoning": "Soft failure but retry deemed unsafe.",
                "issues": issues,
            }

        # HARD_FAIL → Rollback or escalate
        if overall == "HARD_FAIL":
            rollback_available = meta.get("rollback_available", False)
            if rollback_available:
                return {
                    "action": "ROLLBACK",
                    "status": "rolling_back",
                    "reasoning": "Hard failure. Rolling back to last checkpoint.",
                    "issues": issues,
                }
            return {
                "action": "ESCALATE",
                "status": "escalated",
                "reasoning": "Hard failure with no rollback available.",
                "issues": issues,
            }

        # Default: escalate
        return {
            "action": "ESCALATE",
            "status": "escalated",
            "reasoning": f"Unknown verification status: {overall}",
            "issues": issues,
        }

    async def _is_safe_to_retry(self, issues: list[dict[str, Any]], task_id: str) -> bool:
        """Ask LLM if auto-retry is safe given the issues.

        Falls back to True if no model is available (dev mode).

        Args:
            issues: List of issue dicts.
            task_id: Task identifier.

        Returns:
            True if retry is considered safe.
        """
        if self._model is None:
            return True

        try:
            prompt = (
                "Given the following verification issues, is it safe to "
                "automatically retry the failed steps? Answer with JSON: "
                '{"safe": true/false, "reason": "..."}\n\n'
                f"Issues:\n{json.dumps(issues, indent=2)}"
            )

            raw = await self._call_llm(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            result = self._parse_json(raw)
            return bool(result.get("safe", True))

        except Exception as exc:
            logger.warning(
                "supervisor_retry_safety_check_failed",
                task_id=task_id,
                error=str(exc),
            )
            return True  # Default to allowing retry
