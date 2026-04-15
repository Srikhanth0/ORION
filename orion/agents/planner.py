"""PlannerAgent — ReAct-style task decomposition into executable DAGs.

Receives a natural language instruction and produces a structured
execution plan as a TaskDAG-compatible JSON. Uses chain-of-thought
reasoning with tool awareness.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import structlog
from agentscope.message import Msg

from orion.agents.base import BaseOrionAgent
from orion.agentscope_config import build_model

logger = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are ORION's Planner. Decompose the user instruction into a JSON array of subtasks.

CRITICAL: Your ENTIRE response must be a single valid JSON array. No prose, no markdown, \
no code fences. Start with [ and end with ].

Each subtask:
{"id":"t1","description":"...","tool":"list_directory","params":{"path":"."},"depends_on":[]}

AVAILABLE TOOLS - Use these exact names:
- list_directory(path) - List files in a directory
- read_text_file(path) - Read a text file
- write_file(path, content) - Write/create a file
- create_directory(path) - Create a directory
- search_files(path, pattern) - Search for files
- get_file_info(path) - Get file metadata
- move_file(src, dst) - Move/rename a file
- analyze_screen(prompt) - Analyze screen with vision
- click_element(description) - Click UI element
- type_text(text) - Type text
- press_key(key) - Press keyboard key
- browser_navigate(url) - Navigate browser
- browser_click(selector) - Click browser element
- browser_type(text) - Type in browser

Rules:
1. depends_on = list of task ids that must finish first. Use [] for parallel tasks.
2. One tool call per subtask.
3. Use correct tool names ONLY from the list above.
4. Return ONLY the JSON array — no other text whatsoever.
"""


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from model output, tolerating prose/fences."""
    if not text:
        return None
    text = text.strip()

    # Strip markdown blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try to fix common LLM mistakes (like trailing commas before ])
        try:
            fixed = re.sub(r",\s*\]", "]", candidate)
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


class PlannerAgent(BaseOrionAgent):
    def __init__(self, name: str = "Planner", model: Any = None, tool_registry: Any = None) -> None:
        model = model or build_model()
        super().__init__(
            agent_name=name,
            model=model,
            prompt_template=None,
        )
        self._tool_registry = tool_registry
        self.sys_prompt = self._build_sys_prompt()

    def _build_sys_prompt(self) -> str:
        """Compose the system prompt with dynamic tool list."""
        if self._tool_registry and hasattr(self._tool_registry, "describe_all"):
            tool_list = self._tool_registry.describe_all()
        else:
            # Static fallback if registry is unavailable
            tool_list = (
                "- list_directory(path)\n- read_text_file(path)\n"
                "- write_file(path, content)\n- execute_command(command)"
            )

        return f"""\
You are ORION's Planner. Decompose the user instruction into a JSON array of subtasks.

CRITICAL: Your ENTIRE response must be a single valid JSON array. No prose, no markdown, \
no code fences. Start with [ and end with ].

Each subtask:
{{"id":"t1","description":"...","tool":"tool_name","params":{{"arg1":"val1"}},"depends_on":[]}}

AVAILABLE TOOLS - Use these exact names:
{tool_list}

Rules:
1. depends_on = list of task ids that must finish first. Use [] for parallel tasks.
2. One tool call per subtask.
3. Use correct tool names ONLY from the list above.
4. Return ONLY the JSON array — no other text whatsoever.
"""

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

        for attempt in range(3):
            try:
                raw = await self._call_llm(
                    [
                        {"role": "system", "content": self.sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                raw = raw.strip()

                if not raw:
                    logger.warning(
                        "planner_empty_response",
                        attempt=attempt,
                        hint="API key may be rate-limited or quota exhausted",
                    )
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                parsed = _extract_json_array(raw)
                if parsed is not None:
                    return self._make_reply(
                        content=json.dumps({"subtasks": parsed}),
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
                "tool": "execute_command",
                "server": "bash",
                "params": {"command": f"echo 'Task: {instruction}'"},
                "depends_on": [],
            }
        ]
        logger.warning("planner_used_fallback", instruction=instruction[:60])
        return self._make_reply(
            content=json.dumps({"subtasks": fallback}),
            source_msg=x,
            meta_updates={"step_index": 1},
        )
