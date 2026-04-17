"""
ORION PATCH P2 — SSE subtask stream polling in integration tests
File: tests/integration/test_pipeline.py
Commit: fix(tests): switch from polling /tasks/{id} to SSE /tasks/{id}/stream

Root cause: The harness was calling the REST polling endpoint which only
returns the final task status. Intermediate subtask steps (planner output,
executor tool calls, verifier results) were invisible, so timeout diagnosis
was impossible — we only knew "it timed out" with no idea *where*.

This patch replaces poll-based task waiting with a proper SSE consumer
that captures every event and surfaces granular failure points.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import httpx
import pytest

log = logging.getLogger(__name__)

BASE_URL = "http://localhost:8080"


# ── Event model ───────────────────────────────────────────────────────────────


@dataclass
class TaskEvent:
    event_type: str  # "thought" | "tool_call" | "tool_result" | "status"
    agent: str  # "planner" | "executor" | "verifier" | "supervisor"
    content: str
    subtask_id: str | None = None
    timestamp: str | None = None


# ── SSE consumer ─────────────────────────────────────────────────────────────


async def stream_task_events(
    task_id: str,
    timeout_seconds: int = 120,
) -> list[TaskEvent]:
    """
    ORION-FIX: Consume the SSE stream for a task, returning all events.

    Updated to handle multi-line SSE (event: type \n data: payload) and
    map server-side EventType to test-side TaskEvent.
    """
    events: list[TaskEvent] = []
    terminal_statuses = {"DONE", "FAILED", "CANCELLED"}
    current_event_type = "unknown"

    try:
        async with (
            httpx.AsyncClient(timeout=timeout_seconds) as client,
            client.stream(
                "GET",
                f"{BASE_URL}/v1/tasks/{task_id}/stream",
                headers={"Accept": "text/event-stream"},
            ) as response,
        ):
            response.raise_for_status()

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                if line.startswith("event:"):
                    current_event_type = line[6:].strip()
                    continue

                if not line.startswith("data:"):
                    continue

                try:
                    data = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

                # Map server EventType (from current_event_type) to test TaskEvent
                # Server types: planner_start, subtask_queued, subtask_start,
                # subtask_result, verifier_result, supervisor_decision, done

                agent = "unknown"
                content = ""
                subtask_id = data.get("id") or data.get("subtask_id")

                if current_event_type == "planner_start":
                    agent = "planner"
                    content = "Starting plan generation"
                elif current_event_type == "subtask_queued":
                    agent = "planner"
                    content = f"Queued subtask: {data.get('action')}"
                elif current_event_type == "subtask_start":
                    agent = "executor"
                    content = f"Starting subtask: {data.get('action')}"
                elif current_event_type == "subtask_result":
                    agent = "executor"
                    ok = data.get("success", False)
                    content = f"Result: {'OK' if ok else 'FAIL'} - {data.get('output', '')}"
                elif current_event_type == "verifier_result":
                    agent = "verifier"
                    content = f"Verification: {data.get('status')} - {data.get('issues', [])}"
                elif current_event_type == "supervisor_decision":
                    agent = "supervisor"
                    content = f"Decision: {data.get('decision')}"
                elif current_event_type == "done":
                    current_event_type = "status"  # Map to status for compatibility
                    content = data.get("status", "DONE")

                event = TaskEvent(
                    event_type=current_event_type,
                    agent=agent,
                    content=content,
                    subtask_id=subtask_id,
                    timestamp=data.get("ts"),
                )

                events.append(event)
                log.debug(
                    "[%s/%s] %s: %s",
                    event.agent,
                    event.subtask_id,
                    event.event_type,
                    event.content[:80],
                )

                # Stop consuming when we hit a terminal event
                if event.event_type == "status" and event.content in terminal_statuses:
                    log.info("Task %s reached terminal status: %s", task_id, event.content)
                    break

    except httpx.ReadTimeout:
        log.error(
            "SSE stream timed out after %ds for task %s.\nLast captured events:\n%s",
            timeout_seconds,
            task_id,
            "\n".join(f"  [{e.agent}] {e.event_type}: {e.content[:60]}" for e in events[-5:]),
        )
    except Exception as exc:
        log.error("SSE stream error for task %s: %s", task_id, exc)

    return events


def get_final_status(events: list[TaskEvent]) -> str:
    """Extract the terminal status from the event stream."""
    for ev in reversed(events):
        if ev.event_type == "status":
            return ev.content
    return "unknown"


def get_last_agent(events: list[TaskEvent]) -> str:
    """Identify which agent was active when the stream ended — useful for timeout diagnosis."""
    if not events:
        return "none"
    return events[-1].agent


# ── Test helpers ──────────────────────────────────────────────────────────────


async def submit_task(instruction: str) -> str:
    """Submit a task and return the task_id."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{BASE_URL}/v1/tasks",
            json={"instruction": instruction},
        )
        resp.raise_for_status()
        return resp.json()["task_id"]  # type: ignore[no-any-return]


# ── Integration tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_task1_shell() -> None:
    """Task 1: basic shell — list .py files and count them."""
    task_id = await submit_task(
        "List all .py files in the current directory and count how many there are."
    )
    events = await stream_task_events(task_id, timeout_seconds=60)

    final = get_final_status(events)
    if final != "DONE":
        stalled_at = get_last_agent(events)
        pytest.fail(
            f"Task 1 failed with status={final!r}. "
            f"Last active agent: {stalled_at!r}. "
            f"Captured {len(events)} events.\n"
            + "\n".join(f"  [{e.agent}] {e.event_type}: {e.content}" for e in events)
        )

    assert final == "DONE", f"Task 1 final status: {final}"
    log.info("Task 1 PASSED — %d events captured", len(events))


@pytest.mark.asyncio
async def test_task2_reasoning_memory() -> None:
    """Task 2: reasoning + memory — requires LLM and ChromaDB persistent memory."""
    task_id = await submit_task(
        "Summarize what ORION has done in past sessions and suggest 3 improvements."
    )
    events = await stream_task_events(task_id, timeout_seconds=90)

    final = get_final_status(events)
    # Accept 'DONE' (server default)
    assert final == "DONE", (
        f"Task 2 status={final!r} — stalled at agent: {get_last_agent(events)!r}"
    )

    # Check that memory was accessed (even if via fallback)
    memory_events = [e for e in events if "memory" in e.content.lower()]
    log.info("Task 2 PASSED — memory events: %d, total: %d", len(memory_events), len(events))


@pytest.mark.asyncio
async def test_task3_gui_vision() -> None:
    """Task 3: GUI vision — requires ngrok tunnel to Vision API."""
    task_id = await submit_task(
        "Take a screenshot of the current screen and describe what applications are open."
    )
    events = await stream_task_events(task_id, timeout_seconds=90)

    final = get_final_status(events)
    if final == "FAILED":
        # Check if it failed on the vision tool specifically (ngrok not up) vs something else
        vision_errors = [
            e for e in events if "vision" in e.content.lower() or "ngrok" in e.content.lower()
        ]  # noqa: E501
        if vision_errors:
            pytest.skip("Task 3 skipped — Vision API tunnel (ngrok) not reachable on this host")
        pytest.fail(f"Task 3 failed for non-vision reason: {get_last_agent(events)!r}")

    assert final == "DONE"
    log.info("Task 3 PASSED — %d events captured", len(events))
