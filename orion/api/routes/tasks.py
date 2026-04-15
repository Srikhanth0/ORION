"""Task routes — POST /v1/tasks, GET /v1/tasks/{id},
GET /v1/tasks/{id}/stream, POST /v1/tasks/{id}/rollback,
DELETE /v1/tasks/{id}.

Provides fine-grained SSE streaming for every pipeline stage.
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
_MAX_CONCURRENT = int(__import__("os").environ.get("MAX_CONCURRENT_TASKS", "5"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)


def _emit(
    queue: asyncio.Queue[TaskEvent | None] | None,
    event_type: EventType,
    data: dict[str, Any],
) -> None:
    """Push an SSE event to the queue (non-blocking).

    Args:
        queue: Task event queue (may be None if no listener).
        event_type: SSE event type.
        data: Event data payload.
    """
    if queue is None:
        return
    event = TaskEvent(
        event_type=event_type,
        data=data,
        timestamp=datetime.now(tz=UTC),
    )
    try:
        queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.warning("sse_queue_full", event_type=event_type.value)


async def _run_pipeline(
    task_id: str,
    instruction: str,
    context: dict[str, Any],
    timeout: int,
    safe_mode: bool = False,
) -> None:
    """Execute the agent pipeline for a task.

    Emits fine-grained SSE events for each pipeline stage:
    planner_start → subtask_queued → subtask_start → subtask_result →
    verifier_result → supervisor_decision → done.

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

                # Import here to avoid circular deps
                from orion.agentscope_config import init_agentscope

                init_agentscope()

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
                            "safe_mode": safe_mode,
                        }
                    },
                )

                # ── Step 1: Planner ──
                _emit(
                    queue,
                    EventType.PLANNER_START,
                    {
                        "task_id": task_id,
                    },
                )

                from orion.tools.registry import ToolRegistry

                registry_config = (
                    "configs/mcp/sandbox.yaml" if safe_mode else "configs/mcp/servers.yaml"
                )
                registry = ToolRegistry(config_path=registry_config)
                registry.load_from_config()
                # Discover tools for planning
                for cat in registry._servers:
                    await registry.discover_tools(cat)

                planner = PlannerAgent(tool_registry=registry)
                plan_msg = await planner.reply(initial)

                # Emit subtask_queued for each subtask in the plan
                try:
                    plan_data = json.loads(plan_msg.content)
                    subtasks = plan_data.get("subtasks", [])
                    for st in subtasks:
                        _emit(
                            queue,
                            EventType.SUBTASK_QUEUED,
                            {
                                "id": st.get("id", ""),
                                "action": st.get("action", ""),
                                "tool": st.get("tool", ""),
                                "depends_on": st.get("depends_on", []),
                            },
                        )
                except (json.JSONDecodeError, AttributeError):
                    subtasks = []

                _emit(
                    queue,
                    EventType.STEP_DONE,
                    {
                        "step": "planner",
                        "subtask_count": len(subtasks),
                    },
                )

                # ── Step 2: Executor ──
                executor = ExecutorAgent(tool_registry=registry)
                exec_msg = await executor.reply(plan_msg)

                # Emit subtask_result for each result
                try:
                    results = json.loads(exec_msg.content)
                    for r in results:
                        _emit(
                            queue,
                            EventType.SUBTASK_RESULT,
                            {
                                "id": r.get("subtask_id", ""),
                                "success": r.get("ok", False),
                                "output": str(r.get("output", ""))[:200],
                                "duration_ms": r.get("duration_ms", 0),
                            },
                        )
                except (json.JSONDecodeError, AttributeError):
                    results = []

                # ── Step 3: Verifier ──
                verifier = VerifierAgent()
                verify_msg = await verifier.reply(exec_msg)

                try:
                    verify_data = json.loads(verify_msg.content)
                    _emit(
                        queue,
                        EventType.VERIFIER_RESULT,
                        {
                            "status": verify_data.get("status", ""),
                            "pass": verify_data.get("status") == "PASS",
                            "issues": verify_data.get("issues", [])[:5],
                        },
                    )
                except (json.JSONDecodeError, AttributeError):
                    _emit(
                        queue,
                        EventType.VERIFIER_RESULT,
                        {
                            "pass": True,
                            "raw": str(verify_msg.content)[:200],
                        },
                    )

                # ── Step 4: Supervisor ──
                supervisor = SupervisorAgent()
                final_msg = await supervisor.reply(verify_msg)

                try:
                    decision_data = json.loads(final_msg.content)
                    _emit(
                        queue,
                        EventType.SUPERVISOR_DECISION,
                        {
                            "decision": decision_data.get("decision", ""),
                        },
                    )
                except (json.JSONDecodeError, AttributeError):
                    _emit(
                        queue,
                        EventType.SUPERVISOR_DECISION,
                        {
                            "decision": "complete",
                        },
                    )

                _tasks[task_id]["status"] = TaskStatus.DONE
                _tasks[task_id]["result"] = {
                    "output": final_msg.content[:2000],
                }

                _emit(
                    queue,
                    EventType.DONE,
                    {
                        "task_id": task_id,
                        "status": "DONE",
                    },
                )

            finally:
                _semaphore.release()

    except TimeoutError:
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["result"] = {"error": f"Task timed out after {timeout}s"}
    except asyncio.CancelledError:
        _tasks[task_id]["status"] = TaskStatus.CANCELLED
    except Exception as exc:
        logger.error(
            "pipeline_failed",
            task_id=task_id,
            error=str(exc),
        )
        _tasks[task_id]["status"] = TaskStatus.FAILED
        _tasks[task_id]["result"] = {"error": str(exc)[:500]}
    finally:
        if queue:
            await queue.put(None)  # Signal stream end


@router.post(
    "",
    response_model=TaskResponse,
    status_code=202,
)
async def submit_task(body: TaskRequest, request: Request) -> TaskResponse:
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
                detail=(f"Max {_MAX_CONCURRENT} concurrent tasks. Retry later."),
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
            body.safe_mode,
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
async def stream_task(task_id: str, request: Request) -> EventSourceResponse:
    """SSE stream of task events.

    Emits fine-grained events for every pipeline stage:
    planner_start, subtask_queued, subtask_start, subtask_result,
    verifier_result, supervisor_decision, done.
    """
    if task_id not in _tasks:
        raise HTTPException(404, "Task not found")

    queue = _task_queues.get(task_id)
    if queue is None:
        raise HTTPException(404, "Stream not available")

    async def event_generator():
        heartbeat_interval = 15
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=heartbeat_interval)
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


@router.post("/{task_id}/rollback")
async def rollback_task(task_id: str) -> JSONResponse:
    """Trigger rollback for a task.

    Restores all checkpoints in LIFO order.

    Returns:
        JSON with rollback results.
    """
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    try:
        from orion.safety.rollback import RollbackEngine

        engine = RollbackEngine()
        if not engine.has_checkpoints(task_id):
            return JSONResponse(
                status_code=404,
                content={
                    "task_id": task_id,
                    "error": "No checkpoints available",
                },
            )

        results = engine.rollback(task_id)

        logger.info(
            "task_rollback_triggered",
            task_id=task_id,
            steps=len(results),
        )

        return JSONResponse(
            status_code=200,
            content={
                "task_id": task_id,
                "rollback_results": results,
            },
        )

    except Exception as exc:
        logger.error(
            "task_rollback_failed",
            task_id=task_id,
            error=str(exc),
        )
        raise HTTPException(500, f"Rollback failed: {exc}") from exc


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
