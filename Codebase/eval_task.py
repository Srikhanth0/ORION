"""
scripts/eval_task.py — v1.0.1

Fixes:
- RuntimeError('Event loop is closed') on Windows: caused by Python 3.13's
  default ProactorEventLoop closing before httpx async transports finish cleanup.
  Fix: switch to SelectorEventLoop on Windows before anything else runs.
- Each task now runs in its own isolated event loop (no loop reuse between tasks).
"""

from __future__ import annotations

import asyncio
import sys

# ── Windows event loop fix (must be FIRST, before any asyncio import side-effects)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import argparse
import json
import time
from pathlib import Path


SAMPLE_TASKS_PATH = Path("tests/fixtures/sample_tasks.json")


def load_tasks() -> list[dict]:
    with open(SAMPLE_TASKS_PATH) as f:
        return json.load(f)


async def run_single_task(task: dict, verbose: bool) -> dict:
    """Submit one task to the running ORION API and wait for completion."""
    import httpx

    instruction = task["instruction"]
    start = time.monotonic()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Submit
            resp = await client.post(
                "http://localhost:8080/v1/tasks",
                json={"instruction": instruction},
            )
            resp.raise_for_status()
            task_id = resp.json()["task_id"]

            # Poll until done (SSE streaming handled separately for verbose mode)
            for _ in range(120):
                await asyncio.sleep(1)
                status_resp = await client.get(f"http://localhost:8080/v1/tasks/{task_id}")
                if status_resp.status_code == 200:
                    data = status_resp.json()
                    status = data.get("status", "")
                    if status in ("completed", "failed", "partial"):
                        elapsed = time.monotonic() - start
                        return {
                            "instruction": instruction,
                            "status": status,
                            "elapsed": elapsed,
                            "task_id": task_id,
                        }

        return {"instruction": instruction, "status": "timeout", "elapsed": 120.0}

    except httpx.ConnectError:
        return {
            "instruction": instruction,
            "status": "api_unreachable",
            "elapsed": time.monotonic() - start,
            "hint": "Start ORION with: make dev",
        }
    except Exception as exc:
        return {
            "instruction": instruction,
            "status": "error",
            "error": str(exc),
            "elapsed": time.monotonic() - start,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="ORION Evaluation Suite")
    parser.add_argument("--all", action="store_true", help="Run all sample tasks")
    parser.add_argument("--task", type=int, help="Run a specific task by index (1-based)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    tasks = load_tasks()
    print(f"═══ ORION Evaluation Suite ═══")
    print(f"  Loaded {len(tasks)} sample tasks\n")

    if args.task:
        tasks = [tasks[args.task - 1]]

    results = []
    for i, task in enumerate(tasks, 1):
        instruction = task["instruction"]
        print(f"  Running task {i}: {instruction[:60]}...")

        # Each task gets a FRESH event loop — no state leakage between tasks
        result = asyncio.run(run_single_task(task, args.verbose))
        results.append(result)

        status = result["status"]
        elapsed = result.get("elapsed", 0)
        icon = "✓" if status == "completed" else "✗"
        print(f"  [{icon}] {i}: {status.upper()} ({elapsed:.1f}s)")

        if args.verbose and result.get("error"):
            print(f"      Error: {result['error']}")
        if result.get("hint"):
            print(f"      Hint:  {result['hint']}")

    print(f"\n  Results: "
          f"{sum(1 for r in results if r['status'] == 'completed')} passed / "
          f"{len(results)} total")


if __name__ == "__main__":
    main()
