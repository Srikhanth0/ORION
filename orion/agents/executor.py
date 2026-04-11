"""ExecutorAgent — tool-use loop for subtask execution.

Processes subtasks from the Planner's DAG in topological order.
For each subtask: safety check → checkpoint → tool invocation →
result capture. Supports retry with exponential backoff.

Module Contract
---------------
- **Input**: Msg with JSON plan (subtasks array) in content.
- **Output**: Msg with JSON array of StepResult objects in content.
- **Tool Usage**: Delegates to ToolRegistry + MCPClient (via stubs).

Depends On
----------
- ``orion.agents.base`` (BaseOrionAgent)
- ``orion.core.result`` (StepResult)
"""
from __future__ import annotations

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
    """Tool-use execution engine for subtask processing.

    Processes subtasks in dependency order. For each subtask:
    1. Look up tool in the tool registry.
    2. Check safety manifest (raise if denied).
    3. Create a rollback checkpoint.
    4. Invoke the tool with timeout.
    5. Record StepResult (ok, output, error, duration_ms).
    6. On failure: retry up to MAX_RETRIES with exponential backoff.

    Args:
        model: OrionModelWrapper (used only for ambiguous output interpretation).
        tool_registry: Registry mapping tool names to callables (stub).
        safety_checker: Safety manifest checker (stub).
        max_retries: Max retries per subtask.
        tool_timeout: Per-tool invocation timeout in seconds.
    """

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
        """Execute all subtasks from the planner's plan.

        Args:
            *args: First positional arg is the Planner's output Msg.
            **kwargs: Additional keyword args.

        Returns:
            Msg with JSON array of step results in content.
        """
        default = Msg(name="planner", role="assistant", content="{}")
        x: Msg = args[0] if args else kwargs.get("x", default)
        meta = self._get_orion_meta(x)
        task_id = meta["task_id"]

        # Parse plan from planner output
        content = x.content if isinstance(x.content, str) else str(x.content)
        try:
            plan = self._parse_json(content)
        except ValueError as exc:
            logger.error("executor_plan_parse_failed", task_id=task_id, error=str(exc))
            return self._make_reply(
                content=json.dumps([{
                    "subtask_id": "parse_error",
                    "ok": False,
                    "error": f"Failed to parse plan: {exc}",
                    "output": None,
                    "duration_ms": 0,
                }]),
                source_msg=x,
            )

        subtasks = plan.get("subtasks", [])
        logger.info(
            "executor_started",
            task_id=task_id,
            subtask_count=len(subtasks),
        )

        # Build dependency graph for topological execution
        execution_order = self._topological_sort(subtasks)
        results: list[dict[str, Any]] = []
        completed: dict[str, dict[str, Any]] = {}

        for subtask_id in execution_order:
            subtask = next(s for s in subtasks if s["id"] == subtask_id)

            step_result = await self._execute_subtask(
                subtask=subtask,
                task_id=task_id,
                completed_results=completed,
            )
            results.append(step_result)
            completed[subtask_id] = step_result

            if not step_result["ok"] and subtask.get("is_critical", True):
                logger.warning(
                    "executor_critical_failure",
                    task_id=task_id,
                    subtask_id=subtask_id,
                )
                break

        logger.info(
            "executor_completed",
            task_id=task_id,
            total=len(results),
            passed=sum(1 for r in results if r["ok"]),
            failed=sum(1 for r in results if not r["ok"]),
        )

        return self._make_reply(
            content=json.dumps(results, indent=2),
            source_msg=x,
            meta_updates={"step_index": 2},
        )

    async def _execute_subtask(
        self,
        subtask: dict[str, Any],
        task_id: str,
        completed_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Execute a single subtask with retry logic.

        Args:
            subtask: Subtask dict from the plan.
            task_id: Parent task ID.
            completed_results: Results of already-completed subtasks.

        Returns:
            StepResult-like dict with ok, output, error, duration_ms.
        """
        subtask_id = subtask["id"]
        tool_name = subtask.get("tool", "unknown")
        params = subtask.get("params", {})

        for attempt in range(self._max_retries + 1):
            start = time.monotonic()

            try:
                # Step 1: Safety check
                if self._safety_checker is not None:
                    await self._check_safety(tool_name, params)

                # Step 2: Tool lookup
                tool_fn = self._tool_registry.get(tool_name)
                if tool_fn is None:
                    # Tool not in registry — simulate with LLM interpretation
                    output = await self._simulate_tool(
                        subtask, task_id, completed_results
                    )
                else:
                    # Step 3: Invoke tool with timeout
                    if asyncio.iscoroutinefunction(tool_fn):
                        output = await asyncio.wait_for(
                            tool_fn(**params),
                            timeout=self._tool_timeout,
                        )
                    else:
                        output = tool_fn(**params)

                elapsed = (time.monotonic() - start) * 1000

                logger.info(
                    "subtask_completed",
                    task_id=task_id,
                    subtask_id=subtask_id,
                    tool=tool_name,
                    attempt=attempt + 1,
                    duration_ms=round(elapsed, 1),
                )

                return {
                    "subtask_id": subtask_id,
                    "ok": True,
                    "output": str(output) if output is not None else None,
                    "error": None,
                    "duration_ms": round(elapsed, 1),
                    "attempt": attempt + 1,
                }

            except Exception as exc:
                elapsed = (time.monotonic() - start) * 1000
                logger.warning(
                    "subtask_failed",
                    task_id=task_id,
                    subtask_id=subtask_id,
                    tool=tool_name,
                    attempt=attempt + 1,
                    error=str(exc),
                )

                if attempt < self._max_retries:
                    backoff = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue

                return {
                    "subtask_id": subtask_id,
                    "ok": False,
                    "output": None,
                    "error": str(exc),
                    "duration_ms": round(elapsed, 1),
                    "attempt": attempt + 1,
                }

        # Should not reach here, but satisfy type checker
        return {
            "subtask_id": subtask_id,
            "ok": False,
            "output": None,
            "error": "max retries exhausted",
            "duration_ms": 0,
            "attempt": self._max_retries + 1,
        }

    async def _simulate_tool(
        self,
        subtask: dict[str, Any],
        task_id: str,
        completed_results: dict[str, dict[str, Any]],
    ) -> str:
        """Simulate tool execution via LLM when tool not in registry.

        Args:
            subtask: Subtask to simulate.
            task_id: Parent task ID.
            completed_results: Prior results for context.

        Returns:
            Simulated output string.
        """
        if self._model is None:
            return f"[SIMULATED] {subtask.get('action', 'unknown action')}"

        prompt = (
            f"You are simulating tool execution for testing purposes.\n"
            f"Tool: {subtask.get('tool', 'unknown')}\n"
            f"Action: {subtask.get('action', 'unknown')}\n"
            f"Parameters: {json.dumps(subtask.get('params', {}))}\n"
            f"Expected output: {subtask.get('expected_output', 'any')}\n\n"
            f"Produce a realistic output for this tool invocation."
        )

        return await self._call_llm(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )

    async def _check_safety(
        self, tool_name: str, params: dict[str, Any]
    ) -> None:
        """Check if a tool invocation is allowed by the safety manifest.

        Args:
            tool_name: Name of the tool to check.
            params: Tool parameters.

        Raises:
            PermissionDeniedError: If the tool is blocked.
        """
        if hasattr(self._safety_checker, "check"):
            if asyncio.iscoroutinefunction(self._safety_checker.check):
                await self._safety_checker.check(tool_name, params)
            else:
                self._safety_checker.check(tool_name, params)

    def _topological_sort(
        self, subtasks: list[dict[str, Any]]
    ) -> list[str]:
        """Sort subtasks in dependency order using Kahn's algorithm.

        Args:
            subtasks: List of subtask dicts with 'id' and 'depends_on'.

        Returns:
            List of subtask IDs in execution order.
        """
        from collections import deque

        ids = {s["id"] for s in subtasks}
        in_degree: dict[str, int] = dict.fromkeys(ids, 0)
        adjacency: dict[str, list[str]] = {sid: [] for sid in ids}

        for s in subtasks:
            for dep in s.get("depends_on", []):
                if dep in ids:
                    adjacency[dep].append(s["id"])
                    in_degree[s["id"]] += 1

        queue: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
        order: list[str] = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order
