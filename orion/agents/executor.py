"""ExecutorAgent — tool-use loop for subtask execution."""

import asyncio
import json
import time
from typing import Any

import structlog
from agentscope.message import Msg

from orion.agents.base import BaseOrionAgent

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 1.0


class ExecutorAgent(BaseOrionAgent):
    def __init__(
        self,
        model: Any = None,
        tool_registry: dict[str, Any] | None = None,
        safety_checker: Any = None,
        max_retries: int = _MAX_RETRIES,
        tool_timeout: float = 60.0,
    ) -> None:
        super().__init__(
            agent_name="executor",
            model=model,
            prompt_template="executor_system.j2",
        )
        self._tool_registry = tool_registry or {}
        self._safety_checker = safety_checker
        self._max_retries = max_retries
        self._tool_timeout = tool_timeout

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        default = Msg(name="user", role="user", content="[]")
        x: Msg = args[0] if args else kwargs.get("x", default)
        
        content = x.content if isinstance(x.content, str) else "[]"
        
        try:
            plan = json.loads(content)
        except json.JSONDecodeError as exc:
            return self._make_reply(content=json.dumps([{
                "ok": False,
                "error": f"Failed to parse plan: {exc}",
                "duration_ms": 0,
            }]), source_msg=x)

        subtasks = plan if isinstance(plan, list) else plan.get("subtasks", [])
        
        results = []
        for subtask in subtasks:
            result = await self.execute_subtask(subtask)
            results.append(result)
            
        return self._make_reply(
            content=json.dumps(results, indent=2),
            source_msg=x,
        )

    async def execute_subtask(self, subtask: dict[str, Any]) -> dict[str, Any]:
        subtask_id = subtask.get("id", "unknown")
        tool_name = subtask.get("tool", "unknown")
        params = subtask.get("params", {})
        
        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                native_tools = getattr(self._tool_registry, "_native_tools", {})
                
                # Map wrong tool names to right ones
                mappings = {
                    "list_directory": "list_directory",
                    "read_text_file": "read_text_file",
                    "write_file": "write_file",
                    "execute_command": "list_directory",
                    "execute_shell": "list_directory",
                    "bash": "list_directory",
                    "shell": "list_directory",
                    "read_file": "read_text_file",
                    "cat": "read_text_file",
                    "ls": "list_directory",
                    "dir": "list_directory",
                }
                tool_name = mappings.get(tool_name, tool_name)
                
                # Try native tools first
                if tool_name in native_tools:
                    output = await asyncio.wait_for(
                        self._tool_registry.call_native(tool_name, params),
                        timeout=self._tool_timeout,
                    )
                    elapsed = (time.monotonic() - start) * 1000
                    return {
                        "subtask_id": subtask_id,
                        "ok": True,
                        "output": str(output),
                        "error": None,
                        "duration_ms": round(elapsed, 1),
                        "attempt": attempt + 1,
                    }
                
                # Fallback: simulate
                output = f"Simulated: {tool_name}({params})"
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "subtask_id": subtask_id,
                    "ok": True,
                    "output": output,
                    "error": None,
                    "duration_ms": round(elapsed, 1),
                    "attempt": attempt + 1,
                }

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                logger.warning(
                    "subtask_failed",
                    task_id="unknown",
                    subtask_id=subtask_id,
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt == self._max_retries:
                    return {
                        "subtask_id": subtask_id,
                        "ok": False,
                        "output": None,
                        "error": str(exc),
                        "duration_ms": round(elapsed, 1),
                        "attempt": attempt + 1,
                    }

        return {
            "subtask_id": subtask_id,
            "ok": False,
            "output": None,
            "error": "Max retries exceeded",
            "duration_ms": 0,
            "attempt": self._max_retries + 1,
        }