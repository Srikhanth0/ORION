"""VerifierAgent — assertion-based step verification with LLM critique.

Receives step results from the Executor. Performs deterministic
assertions first (string match, regex, JSON path), then calls the
LLM for a holistic critique. Produces a VerificationReport.

Module Contract
---------------
- **Input**: Msg with JSON step results array from Executor.
- **Output**: Msg with VerificationReport JSON.
- **State**: Read-only — Verifier never mutates task state.

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


class VerifierAgent(BaseOrionAgent):
    """Assertion-based verifier with LLM critique.

    For each step result:
    1. Run deterministic assertions (string match, regex, JSON path).
    2. Collect assertion results.
    3. Call LLM with critique prompt including all results.
    4. Produce VerificationReport with PASS/SOFT_FAIL/HARD_FAIL.

    The Verifier never mutates state — it only reads and reports.

    Args:
        model: OrionModelWrapper for LLM critique.
        original_instruction: The original task instruction for context.
    """

    def __init__(
        self,
        model: Any = None,
        original_instruction: str = "",
    ) -> None:
        super().__init__(
            agent_name="verifier",
            model=model,
            prompt_template="verifier_system.j2",
        )
        self._original_instruction = original_instruction

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        """Verify execution results and produce a VerificationReport.

        Args:
            *args: First positional arg is the Executor's output Msg.
            **kwargs: Additional keyword args.

        Returns:
            Msg with VerificationReport JSON in content.
        """
        default = Msg(name="executor", role="assistant", content="[]")
        x: Msg = args[0] if args else kwargs.get("x", default)
        meta = self._get_orion_meta(x)
        task_id = meta["task_id"]

        content = x.content if isinstance(x.content, str) else str(x.content)
        try:
            step_results = json.loads(content)
        except json.JSONDecodeError:
            step_results = []

        logger.info(
            "verifier_started",
            task_id=task_id,
            step_count=len(step_results),
        )

        # Phase 1: Deterministic assertions
        assertion_results = self._run_assertions(step_results)

        # Phase 2: LLM critique
        report = await self._llm_critique(
            task_id=task_id,
            step_results=step_results,
            assertion_results=assertion_results,
        )

        logger.info(
            "verifier_completed",
            task_id=task_id,
            overall=report["overall"],
            recommendation=report["recommendation"],
            issues=len(report.get("issues", [])),
        )

        return self._make_reply(
            content=json.dumps(report, indent=2),
            source_msg=x,
            meta_updates={"step_index": 3},
        )

    def _run_assertions(self, step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run deterministic assertions on each step result.

        Args:
            step_results: List of StepResult-like dicts.

        Returns:
            List of assertion result dicts.
        """
        assertions = []

        for result in step_results:
            subtask_id = result.get("subtask_id", "unknown")
            ok = result.get("ok", False)
            output = result.get("output", "")
            expected = result.get("expected_output", "")

            assertion = {
                "subtask_id": subtask_id,
                "execution_ok": ok,
                "checks": [],
            }

            if not ok:
                assertion["checks"].append(
                    {
                        "type": "execution_status",
                        "passed": False,
                        "detail": f"Subtask failed: {result.get('error', 'unknown')}",
                    }
                )
            else:
                # String containment check
                if expected and output:
                    contains = expected.lower() in str(output).lower()
                    assertion["checks"].append(
                        {
                            "type": "string_contains",
                            "passed": contains,
                            "detail": (f"Expected output to contain '{expected[:50]}'"),
                        }
                    )

                # Non-empty output check
                assertion["checks"].append(
                    {
                        "type": "non_empty_output",
                        "passed": bool(output and str(output).strip()),
                        "detail": "Output should not be empty",
                    }
                )

            assertions.append(assertion)

        return assertions

    async def _llm_critique(
        self,
        task_id: str,
        step_results: list[dict[str, Any]],
        assertion_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call LLM for holistic critique of execution results.

        Args:
            task_id: Task identifier.
            step_results: Raw execution results.
            assertion_results: Deterministic assertion results.

        Returns:
            VerificationReport dict.
        """
        # If no model, produce report from assertions alone
        if self._model is None:
            return self._report_from_assertions(assertion_results)

        system_prompt = self._render_prompt(
            system_context="ORION Verifier Agent",
            available_tools="N/A (verification only)",
            recent_memory="",
            task=self._original_instruction or "Task instruction unavailable",
            output_format=self._get_report_format(),
        )

        user_content = (
            f"## Original Task\n{self._original_instruction}\n\n"
            f"## Execution Results\n{json.dumps(step_results, indent=2)}\n\n"
            f"## Assertion Results\n{json.dumps(assertion_results, indent=2)}\n\n"
            "Based on the above, produce a VerificationReport."
        )

        try:
            raw = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            report = self._parse_json(raw)
            # Validate required fields
            if "overall" not in report:
                report["overall"] = "SOFT_FAIL"
            if "recommendation" not in report:
                report["recommendation"] = "RETRY_STEP"
            if "issues" not in report:
                report["issues"] = []
            return report

        except (ValueError, Exception) as exc:
            logger.warning(
                "verifier_llm_critique_failed",
                task_id=task_id,
                error=str(exc),
            )
            return self._report_from_assertions(assertion_results)

    def _report_from_assertions(self, assertion_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a report purely from deterministic assertions.

        Args:
            assertion_results: Assertion results list.

        Returns:
            VerificationReport dict.
        """
        all_passed = True
        has_failures = False
        issues = []

        for ar in assertion_results:
            for check in ar.get("checks", []):
                if not check.get("passed", False):
                    all_passed = False
                    if check.get("type") == "execution_status":
                        has_failures = True
                    issues.append(
                        {
                            "subtask_id": ar["subtask_id"],
                            "check_type": check["type"],
                            "detail": check.get("detail", ""),
                        }
                    )

        if all_passed:
            overall = "PASS"
            recommendation = "DONE"
        elif has_failures:
            overall = "HARD_FAIL"
            recommendation = "ROLLBACK"
        else:
            overall = "SOFT_FAIL"
            recommendation = "RETRY_STEP"

        return {
            "overall": overall,
            "issues": issues,
            "recommendation": recommendation,
        }

    def _get_report_format(self) -> str:
        """Return the VerificationReport JSON schema."""
        return """\
{
  "overall": "PASS | SOFT_FAIL | HARD_FAIL",
  "issues": [
    {
      "subtask_id": "string",
      "check_type": "string",
      "detail": "string"
    }
  ],
  "recommendation": "DONE | RETRY_STEP | ESCALATE | ROLLBACK"
}"""
