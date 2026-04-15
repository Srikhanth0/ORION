"""
orion/agents/planner.py  — v1.0.1 patch

Root causes fixed:
1. AgentScope model() returns a ModelResponse; .text can be None/'' on
   free-tier empty completions or rate-limit silent failures.
2. Old code retried with same broken prompt. Now enforces JSON on every call
   and uses a fence-stripping extractor that finds the [ ... ] block.
3. Windows ProactorEventLoop / httpx cleanup RuntimeError is separate —
   see scripts/eval_task.py fix below.
"""

from __future__ import annotations

import json
import re
import time

from agentscope.agents import AgentBase
from agentscope.message import Msg

from orion.llm.router import get_model_config

_SYSTEM_PROMPT = """\
You are ORION's Planner. Decompose the user instruction into a JSON array of subtasks.

CRITICAL: Your ENTIRE response must be a single valid JSON array. No prose, no markdown, \
no code fences. Start with [ and end with ].

Each subtask:
{"id":"t1","description":"...","tool":"execute_command","server":"bash",\
"arguments":{"command":"..."},"depends_on":[]}

Available tools:
- bash server: execute_command(command)
- filesystem server: read_file(path), write_file(path,content), list_directory(path)
- browser server: navigate(url), click(selector), type(text), get_page_content()
- github server: create_issue, list_repos, get_file_content
- screen server: analyze_screen(), click_element(description), type_text(text)

Rules:
1. depends_on = list of task ids that must finish first. Use [] for parallel tasks.
2. One tool call per subtask.
3. Return ONLY the JSON array — no other text whatsoever.
"""


def _extract_json_array(text: str) -> list | None:
    """Extract a JSON array from model output, tolerating prose/fences."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


class PlannerAgent(AgentBase):
    def __init__(self, name: str = "Planner") -> None:
        super().__init__(
            name=name,
            sys_prompt=_SYSTEM_PROMPT,
            model_config_name=get_model_config("planner"),
        )

    def reply(self, x: Msg) -> Msg:
        import structlog
        log = structlog.get_logger()

        content = x.content if isinstance(x.content, dict) else {"instruction": str(x.content)}
        instruction = content.get("instruction", "")
        retry_feedback = content.get("retry_feedback", "")
        retry_count = content.get("retry_count", 0)

        prompt = f"Instruction: {instruction}"
        if retry_feedback:
            prompt += f"\n\nPrevious attempt #{retry_count} failed: {retry_feedback}\nAdjust the plan."
        prompt += "\n\nRespond with ONLY the JSON array:"

        for attempt in range(3):
            try:
                response = self.model(
                    [
                        {"role": "system", "content": self.sys_prompt},
                        {"role": "user", "content": prompt},
                    ]
                )
                raw = (response.text or "").strip()
                if not raw:
                    log.warning(
                        "planner_empty_response",
                        attempt=attempt,
                        hint="API key may be rate-limited or quota exhausted",
                    )
                    time.sleep(1.5 * (attempt + 1))
                    continue

                parsed = _extract_json_array(raw)
                if parsed is not None:
                    return Msg(name=self.name, content=parsed, role="assistant")

                log.warning(
                    "json_parse_failed",
                    agent="planner",
                    error="No JSON array found in output",
                    text_preview=raw[:120],
                )
            except Exception as exc:
                log.warning("planner_model_error", attempt=attempt, error=str(exc))
                time.sleep(1.5 * (attempt + 1))

        # Fallback: single echo so pipeline doesn't hang
        fallback = [{
            "id": "t1",
            "description": instruction,
            "tool": "execute_command",
            "server": "bash",
            "arguments": {"command": f"echo 'Task: {instruction}'"},
            "depends_on": [],
        }]
        log.warning("planner_used_fallback", instruction=instruction[:60])
        return Msg(name=self.name, content=fallback, role="assistant")
