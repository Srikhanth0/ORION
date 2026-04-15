"""ORION task dispatcher — DAG-based parallel subtask execution.

Entry point for external callers (API, CLI). Accepts a natural language
instruction, dispatches through the HiClaw pipeline with DAG-based
parallel execution for independent subtasks.

Module Contract
---------------
- **Inputs**: Task instruction string + optional config.
- **Outputs**: ``Msg`` containing ``TaskResult`` in metadata.

Depends On
----------
- ``orion.orchestrator.pipeline`` (OrionPipeline)
- ``orion.agents.executor`` (ExecutorAgent, topological_sort)
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog
from agentscope.message import Msg

from orion.core.exceptions import OrionError

logger = structlog.get_logger(__name__)


async def execute_dag(
    subtasks: list[dict[str, Any]],
    executor: Any,
    task_id: str,
    completed: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Execute subtasks respecting dependency DAG with parallel batches.

    Groups subtasks by dependency layer. Independent subtasks in the
    same layer execute concurrently via ``asyncio.gather``. Tasks with
    unsatisfied dependencies wait until their predecessors complete.

    Args:
        subtasks: List of subtask dicts with 'id' and 'depends_on'.
        executor: ExecutorAgent instance with ``execute_subtask`` method.
        task_id: Parent task ID for logging.
        completed: Optional pre-populated results dict.

    Returns:
        Dict mapping subtask IDs to their result dicts.

    Raises:
        RuntimeError: If the DAG has unsatisfiable dependencies (cycle).
    """
    results: dict[str, dict[str, Any]] = dict(completed or {})
    pending = {t["id"]: t for t in subtasks}

    batch_num = 0
    while pending:
        # Find tasks whose dependencies are all resolved
        ready = [
            t for t in pending.values() if all(dep in results for dep in t.get("depends_on", []))
        ]

        if not ready:
            remaining = list(pending.keys())
            raise RuntimeError(f"DAG has unsatisfiable dependencies. Stuck tasks: {remaining}")

        batch_num += 1
        ready_ids = [t["id"] for t in ready]
        logger.info(
            "dag_batch_start",
            task_id=task_id,
            batch=batch_num,
            parallel_count=len(ready),
            subtask_ids=ready_ids,
        )

        # Execute ready batch in parallel
        coros = [
            executor.execute_subtask(
                subtask=t,
                task_id=task_id,
                completed_results=results,
            )
            for t in ready
        ]
        batch_results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(ready, batch_results, strict=True):
            if isinstance(result, BaseException):
                results[task["id"]] = {
                    "subtask_id": task["id"],
                    "ok": False,
                    "output": None,
                    "error": str(result),
                    "duration_ms": 0,
                    "attempt": 1,
                }
                logger.error(
                    "dag_subtask_exception",
                    task_id=task_id,
                    subtask_id=task["id"],
                    error=str(result),
                )
            else:
                results[task["id"]] = result

            del pending[task["id"]]

        logger.info(
            "dag_batch_complete",
            task_id=task_id,
            batch=batch_num,
            completed_ids=ready_ids,
            remaining=len(pending),
        )

    return results


class TaskDispatcher:
    """Routes incoming task instructions to the HiClaw pipeline.

    Handles task timeout enforcement (default 300s) and top-level
    error handling. Can be used as the main entry point from the
    API server or CLI.

    Args:
        pipeline: Configured OrionPipeline instance.
        global_timeout: Maximum seconds for any single task.
    """

    def __init__(
        self,
        pipeline: Any,
        global_timeout: float = 300.0,
    ) -> None:
        self._pipeline = pipeline
        self._global_timeout = global_timeout

    async def dispatch(
        self,
        instruction: str,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Msg:
        """Dispatch a task through the HiClaw pipeline with timeout.

        Args:
            instruction: Natural language task instruction.
            task_id: Optional task ID.
            context: Optional OS context.

        Returns:
            Final Msg from the Supervisor.

        Raises:
            OrionError: On pipeline failure.
            asyncio.TimeoutError: If task exceeds global timeout.
        """
        logger.info(
            "task_dispatched",
            instruction=instruction[:100],
            task_id=task_id,
            timeout=self._global_timeout,
        )

        try:
            result = await asyncio.wait_for(
                self._pipeline.run(
                    instruction=instruction,
                    task_id=task_id,
                    context=context,
                ),
                timeout=self._global_timeout,
            )
            logger.info(
                "task_completed",
                task_id=task_id,
            )
            return result

        except TimeoutError:
            logger.error(
                "task_timeout",
                task_id=task_id,
                timeout=self._global_timeout,
            )
            raise

        except OrionError:
            raise

        except Exception as exc:
            logger.error(
                "task_unexpected_error",
                task_id=task_id,
                error=str(exc),
            )
            raise OrionError(
                f"Unexpected error during task dispatch: {exc}",
                task_id=task_id,
            ) from exc
