"""ORION task dispatcher — routes tasks to the appropriate pipeline.

Entry point for external callers (API, CLI). Accepts a natural language
instruction, instantiates or retrieves the HiClaw pipeline, and returns
the final TaskResult.

Module Contract
---------------
- **Inputs**: Task instruction string + optional config.
- **Outputs**: ``Msg`` containing ``TaskResult`` in metadata.

Depends On
----------
- ``orion.orchestrator.pipeline`` (OrionPipeline)
"""
from __future__ import annotations

import asyncio
from typing import Any

import structlog
from agentscope.message import Msg

from orion.core.exceptions import OrionError

logger = structlog.get_logger(__name__)


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
