"""PlannerAgent — ReAct-style task decomposition into executable DAGs.

Receives a natural language instruction and produces a structured
execution plan as a TaskDAG-compatible JSON. Uses chain-of-thought
reasoning with tool awareness.

Module Contract
---------------
- **Input**: Msg with task instruction in content, orion_meta in metadata.
- **Output**: Msg with JSON plan in content (subtasks array with depends_on).
- **LLM Usage**: Full ReAct prompting via planner_system.j2.

Depends On
----------
- ``orion.agents.base`` (BaseOrionAgent)
- ``orion.core.task`` (Subtask, TaskDAG)
"""
from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import structlog
from agentscope.message import Msg

from orion.agents.base import BaseOrionAgent
from orion.core.exceptions import PlanError

logger = structlog.get_logger(__name__)


class PlannerAgent(BaseOrionAgent):
    """ReAct-style planner that decomposes tasks into executable DAGs.

    Uses chain-of-thought reasoning to produce structured JSON plans.
    Each plan includes a reasoning chain and an array of subtasks with
    tool bindings, parameters, dependencies, and destructiveness flags.

    Args:
        model: OrionModelWrapper instance.
        available_tools: List of tool descriptions for the prompt.
        max_subtasks: Maximum subtasks per plan (guard rail).
        max_retries: Maximum LLM retries on parse failure.
    """

    def __init__(
        self,
        model: Any = None,
        available_tools: list[dict[str, Any]] | None = None,
        max_subtasks: int = 20,
        max_retries: int = 2,
    ) -> None:
        super().__init__(
            agent_name="planner",
            model=model,
            prompt_template="planner_system.j2",
        )
        self._available_tools = available_tools or []
        self._max_subtasks = max_subtasks
        self._max_retries = max_retries

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        """Generate an execution plan from a task instruction.

        Args:
            *args: First positional arg should be the input Msg.
            **kwargs: Additional keyword args.

        Returns:
            Msg with structured JSON plan in content.

        Raises:
            PlanError: If planning fails after retries.
        """
        default = Msg(name="user", role="user", content="")
        x: Msg = args[0] if args else kwargs.get("x", default)
        meta = self._get_orion_meta(x)
        task_id = meta["task_id"]
        instruction = x.content if isinstance(x.content, str) else str(x.content)

        logger.info(
            "planner_started",
            task_id=task_id,
            instruction=instruction[:100],
        )

        # Build system prompt
        system_prompt = self._render_prompt(
            system_context=self._get_system_context(),
            available_tools=json.dumps(self._available_tools, indent=2),
            recent_memory="No prior plans available.",
            task=instruction,
            output_format=self._get_output_format(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]

        # Attempt planning with retries
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = await self._call_llm(
                    messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )

                plan = self._parse_json(raw)
                self._validate_plan(plan)

                logger.info(
                    "planner_completed",
                    task_id=task_id,
                    subtask_count=len(plan.get("subtasks", [])),
                    attempt=attempt + 1,
                )

                return self._make_reply(
                    content=json.dumps(plan, indent=2),
                    source_msg=x,
                    meta_updates={"step_index": 1},
                )

            except (ValueError, PlanError) as exc:
                last_error = str(exc)
                logger.warning(
                    "planner_retry",
                    task_id=task_id,
                    attempt=attempt + 1,
                    error=last_error,
                )
                # Add error feedback for retry
                messages.append({"role": "assistant", "content": raw if "raw" in dir() else ""})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Your plan was invalid: {last_error}\n"
                        "Please fix the JSON and try again."
                    ),
                })

        raise PlanError(
            f"Planner failed after {self._max_retries + 1} attempts: {last_error}",
            task_id=task_id,
            raw_output=last_error,
        )

    def _validate_plan(self, plan: dict[str, Any]) -> None:
        """Validate a parsed plan structure.

        Args:
            plan: Parsed JSON plan dict.

        Raises:
            PlanError: If the plan structure is invalid.
        """
        if "subtasks" not in plan:
            raise PlanError("Plan missing 'subtasks' array")

        subtasks = plan["subtasks"]
        if not isinstance(subtasks, list):
            raise PlanError("'subtasks' must be an array")

        if len(subtasks) == 0:
            raise PlanError("Plan has zero subtasks")

        if len(subtasks) > self._max_subtasks:
            raise PlanError(
                f"Plan has {len(subtasks)} subtasks, max is {self._max_subtasks}"
            )

        ids = set()
        for st in subtasks:
            if "id" not in st or "action" not in st:
                raise PlanError(f"Subtask missing required fields: {st}")
            if st["id"] in ids:
                raise PlanError(f"Duplicate subtask ID: {st['id']}")
            ids.add(st["id"])

        # Check dependency references
        for st in subtasks:
            for dep in st.get("depends_on", []):
                if dep not in ids:
                    raise PlanError(
                        f"Subtask '{st['id']}' depends on unknown ID '{dep}'"
                    )

    def _get_system_context(self) -> str:
        """Build OS context string for the prompt."""
        return (
            f"OS: {platform.system()} {platform.release()}\n"
            f"Hostname: {platform.node()}\n"
            f"CWD: {Path.cwd()}\n"
            f"User: ORION Agent"
        )

    def _get_output_format(self) -> str:
        """Return the expected output JSON schema for the prompt."""
        return """\
{
  "chain_of_thought": "string — your step-by-step reasoning",
  "subtasks": [
    {
      "id": "s1",
      "action": "human-readable description",
      "tool": "exact_tool_name_from_registry",
      "params": {},
      "expected_output": "what success looks like",
      "depends_on": [],
      "is_destructive": false
    }
  ]
}"""
