"""Task routes — POST /v1/tasks, GET /v1/tasks/{id},
GET /v1/tasks/{id}/stream, DELETE /v1/tasks/{id}.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from orion.api.schemas import (
    EventType,
    ProblemDetail,
    TaskDetailResponse,
    TaskEvent,
    TaskRequest,
    TaskResponse,
    TaskStatus,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/tasks", tags=["tasks"])

# In-memory task store (replaced by DB in production)
_tasks: dict[str, dict[str, Any]] = {}
_task_queues: dict[str, asyncio.Queue[TaskEvent | None]] = {}
_task_handles: dict[str, asyncio.Task[Any]] = {}

# Concurrency limiter
_MAX_CONCURRENT = int(
    __import__("os").environ.get("MAX_CONCURRENT_TASKS", "5")
)
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


async def _run_pipeline(
    task_id: str,
    instruction: str,
    context: dict[str, Any],
    timeout: int,
) -> None:
    """Execute the agent pipeline for a task.

    Args:
        task_id: Task ID.
        instruction: Task instruction.
        context: OS context.
        timeout: Timeout in seconds.
    """
    queue = _task_queues.get(task_id)

    try:
        async with asyncio.timeout(timeout):
            acquired = _semaphore.acquire()
            with contextlib.suppress(TypeError):
                await acquired

            try:
                _tasks[task_id]["status"] = TaskStatus.RUNNING

                if queue:
                    await queue.put(TaskEvent(
                        event_type=EventType.STEP_START,
                        data={"step": "pipeline_start"},
                        timestamp=datetime.now(tz=UTC),
                    ))

                # Import here to avoid circular deps
                from agentscope.message import Msg

                from orion.agents.executor import (
                    ExecutorAgent,
                )
                from orion.agents.planner import PlannerAgent
                from orion.agents.supervisor import (
                    SupervisorAgent,
                )
                from orion.agents.verifier import (
                    VerifierAgent,
                )

                initial = Msg(
                    name="user",
                    role="user",
                    content=instruction,
                    metadata={
                        "orion_meta": {
                            "task_id": task_id,
                            "step_index": 0,
                            "retry_count": 0,
                            "rollback_available": False,
                            "trace_id": task_id,
                            "context": context,
                        }
                    },
                )

                planner = PlannerAgent()
                plan_msg = await planner.reply(initial)

                if queue:
                    await queue.put(TaskEvent(
                        event_type=EventType.STEP_DONE,
                        data={
                            "step": "planner",
                            "plan": plan_msg.content[:500],
                        },
                        timestamp=datetime.now(tz=UTC),
                    ))

                executor = ExecutorAgent()
                exec_msg = await executor.reply(plan_msg)

                if queue:
                    await queue.put(TaskEvent(
                        event_type=EventType.STEP_DONE,
                        data={"step": "executor"},
                        timestamp=datetime.now(tz=UTC),
                    ))

                verifier = VerifierAgent()
                verify_msg = await verifier.reply(exec_msg)

                supervisor = SupervisorAgent()
                final_msg = await supervisor.reply(
                    verify_msg
                )

                _tasks[task_id]["status"] = TaskStatus.DONE
                _tasks[task_id]["result"] = {
                    "output": final_msg.content[:2000],
                }

                if queue:
                    await queue.put(TaskEvent(
                        event_type=EventType.TASK_DONE,
                        data={
                            "result": final_msg.content[:500]
                        },
                        timestamp=datetime.now(tz=UTC),
                    ))

            finally:
                _semaphore.release()

    except TimeoutError:
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["result"] = {
            "error": f"Task timed out after {timeout}s"
        }
    except asyncio.CancelledError:
        _tasks[task_id]["status"] = TaskStatus.CANCELLED
    except Exception as exc:
        logger.error(
            "pipeline_failed",
            task_id=task_id,
            error=str(exc),
        )
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["result"] = {
            "error": str(exc)[:500]
        }
    finally:
        if queue:
            await queue.put(None)  # Signal stream end


@router.post(
    "",
    response_model=TaskResponse,
    status_code=202,
)
async def submit_task(
    body: TaskRequest, request: Request
) -> TaskResponse:
    """Submit a new task for async execution.

    Returns 202 Accepted with task_id for polling.
    Returns 429 if concurrency limit is exceeded.
    """
    # Check semaphore without blocking
    if _semaphore._value == 0:
        return JSONResponse(
            status_code=429,
            content=ProblemDetail(
                type="/errors/too-many-tasks",
                title="Too Many Tasks",
                status=429,
                detail=(
                    f"Max {_MAX_CONCURRENT} concurrent "
                    "tasks. Retry later."
                ),
            ).model_dump(),
            headers={"Retry-After": "30"},
        )

    task_id = f"task_{uuid.uuid4().hex[:12]}"
    now = datetime.now(tz=UTC)

    _tasks[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.QUEUED,
        "created_at": now,
        "instruction": body.instruction,
        "result": None,
        "steps": [],
        "cost_usd": None,
    }

    _task_queues[task_id] = asyncio.Queue()

    handle = asyncio.create_task(
        _run_pipeline(
            task_id,
            body.instruction,
            body.context,
            body.timeout_seconds,
        )
    )
    _task_handles[task_id] = handle

    logger.info(
        "task_submitted",
        task_id=task_id,
        instruction=body.instruction[:100],
    )

    return TaskResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        created_at=now,
        estimated_seconds=body.timeout_seconds,
    )


@router.get(
    "/{task_id}",
    response_model=TaskDetailResponse,
)
async def get_task(task_id: str) -> TaskDetailResponse:
    """Get task status and result."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="/errors/task-not-found",
                title="Task Not Found",
                status=404,
                detail=f"Task '{task_id}' not found",
                task_id=task_id,
            ).model_dump(),
        )

    return TaskDetailResponse(
        task_id=task["task_id"],
        status=task["status"],
        created_at=task["created_at"],
        result=task.get("result"),
        steps=[],
        cost_usd=task.get("cost_usd"),
    )


@router.get("/{task_id}/stream")
async def stream_task(
    task_id: str, request: Request
) -> EventSourceResponse:
    """SSE stream of task events."""
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")

    queue = _task_queues.get(task_id)
    if queue is None:
        raise HTTPException(
            404, "Stream not available"
        )

    async def event_generator():
        heartbeat_interval = 15
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=heartbeat_interval
                )
            except TimeoutError:
                yield {"comment": "keep-alive"}
                continue

            if event is None:
                break

            yield {
                "event": event.event_type.value,
                "data": json.dumps(event.data),
            }

    return EventSourceResponse(event_generator())


@router.delete("/{task_id}", status_code=204)
async def cancel_task(task_id: str) -> None:
    """Cancel a running task."""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    handle = _task_handles.get(task_id)
    if handle and not handle.done():
        handle.cancel()
        task["status"] = TaskStatus.CANCELLED

    logger.info("task_cancelled", task_id=task_id)
