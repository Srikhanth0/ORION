"""PlannerAgent — ReAct-style task decomposition into executable DAGs.

Receives a natural language instruction and produces a structured
execution plan as a TaskDAG-compatible JSON. Uses chain-of-thought
reasoning with tool awareness.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from agentscope.message import Msg

from orion.agents.base import BaseOrionAgent
from orion.core.utils.json_utils import extract_json_array

logger = structlog.get_logger(__name__)

# NOTE: System prompt is built dynamically by _build_sys_prompt() using the
# live tool registry. No static _SYSTEM_PROMPT needed.


class PlannerAgent(BaseOrionAgent):
    def __init__(
        self,
        name: str = "Planner",
        model: Any = None,
        tool_registry: Any = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            agent_name=name,
            model=model,
            prompt_template="planner_system.j2",
        )
        self._tool_registry = tool_registry
        self._max_retries = max_retries

    def _build_sys_prompt(self, instruction: str) -> str:
        """Compose the system prompt with dynamic tool list."""
        if self._tool_registry and hasattr(self._tool_registry, "describe_all"):
            tool_list = self._tool_registry.describe_all()
        else:
            # Static fallback if registry is unavailable
            tool_list = (
                "- list_directory(path)\n- read_text_file(path)\n"
                "- write_file(path, content)\n- execute_command(command)"
            )

        return self._render_prompt(
            system_context="You are ORION's Planner engine.",
            available_tools=tool_list,
            recent_memory="No recent memory available.",
            task=instruction,
            output_format="A JSON object with 'chain_of_thought' and 'subtasks' array.",
        )

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        default = Msg(name="user", role="user", content="{}")
        x: Msg = args[0] if args else kwargs.get("x", default)

        content = x.content if isinstance(x.content, dict) else {"instruction": str(x.content)}
        instruction = content.get("instruction", "")
        retry_feedback = content.get("retry_feedback", "")
        retry_count = content.get("retry_count", 0)

        prompt = f"Instruction: {instruction}"
        if retry_feedback:
            prompt += (
                f"\n\nPrevious attempt #{retry_count} failed: {retry_feedback}\nAdjust the plan."  # noqa: E501
            )
        prompt += "\n\nRespond with ONLY the JSON array:"

        sys_prompt = self._build_sys_prompt(instruction)

        for attempt in range(self._max_retries):
            try:
                raw = await self._call_llm(
                    [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                raw = raw.strip()

                if not raw:
                    logger.warning("planner_empty_response", attempt=attempt)
                    await asyncio.sleep(1.0)
                    continue

                # The template expects a JSON object with 'subtasks'
                try:
                    parsed_obj = self._parse_json(raw)
                    subtasks = parsed_obj.get("subtasks", [])
                except Exception:
                    # Fallback to array extraction if object parsing fails
                    subtasks = extract_json_array(raw)

                if subtasks:
                    return self._make_reply(
                        content=json.dumps({"subtasks": subtasks}),
                        source_msg=x,
                        meta_updates={"step_index": 1},
                    )

                logger.warning(
                    "json_parse_failed",
                    agent="planner",
                    error="No JSON array found in output",
                    text_preview=raw[:120],
                )
            except Exception as exc:
                logger.warning("planner_model_error", attempt=attempt, error=str(exc))
                await asyncio.sleep(1.5 * (attempt + 1))

        fallback = [
            {
                "id": "t1",
                "description": instruction,
                "tool": "list_directory",
                "params": {"path": "."},
                "depends_on": [],
            }
        ]
        logger.warning("planner_used_fallback", instruction=instruction[:60])
        return self._make_reply(
            content=json.dumps({"subtasks": fallback}),
            source_msg=x,
            meta_updates={"step_index": 1},
        )
