#!/usr/bin/env python3
"""ORION eval_task — end-to-end evaluation suite.

Runs sample tasks from fixtures, measures performance, and
generates a JSON evaluation report.

Usage:
    python scripts/eval_task.py --all
    python scripts/eval_task.py --task 1
    python scripts/eval_task.py --task 9 --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "sample_tasks.json"
)


def load_tasks() -> list[dict]:
    """Load sample tasks from fixture."""
    with open(_FIXTURES) as f:
        return json.load(f)


def run_task(task: dict, verbose: bool = False) -> dict:
    """Run a single evaluation task."""
    task_id = task.get("id", "unknown")
    instruction = task.get("instruction", "")
    expected = task.get("expected_status", "DONE")

    if verbose:
        print(f"  Running task {task_id}: {instruction[:60]}...")

    start = time.monotonic()

    try:
        from agentscope.message import Msg

        from orion.agents.executor import ExecutorAgent
        from orion.agents.planner import PlannerAgent
        from orion.agents.supervisor import SupervisorAgent
        from orion.agents.verifier import VerifierAgent

        async def _run() -> dict:
            initial = Msg(
                name="user", role="user",
                content=instruction,
                metadata={"orion_meta": {
                    "task_id": f"eval_{task_id}",
                    "step_index": 0, "retry_count": 0,
                    "rollback_available": False,
                    "trace_id": f"eval_{task_id}",
                    "context": task.get("context", {}),
                }},
            )
            planner = PlannerAgent()
            plan_msg = await planner.reply(initial)
            executor = ExecutorAgent()
            exec_msg = await executor.reply(plan_msg)
            verifier = VerifierAgent()
            verify_msg = await verifier.reply(exec_msg)
            supervisor = SupervisorAgent()
            final_msg = await supervisor.reply(verify_msg)
            return {"result": final_msg.content[:200]}

        asyncio.run(_run())
        status = "DONE"
        error = None
    except Exception as exc:
        status = "FAILED"
        error = str(exc)[:200]

    duration = time.monotonic() - start
    match = status == expected

    if verbose:
        mark = "✓" if match else "✗"
        print(f"  [{mark}] {task_id}: {status} ({duration:.1f}s)")

    return {
        "task_id": task_id,
        "instruction": instruction[:100],
        "status": status,
        "expected": expected,
        "match": match,
        "duration_seconds": round(duration, 2),
        "error": error,
    }


def main() -> int:
    """Run evaluation suite."""
    parser = argparse.ArgumentParser(description="ORION eval suite")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--task", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    tasks = load_tasks()
    print("═══ ORION Evaluation Suite ═══")
    print(f"  Loaded {len(tasks)} sample tasks\n")

    if args.task is not None:
        idx = args.task - 1
        if 0 <= idx < len(tasks):
            tasks = [tasks[idx]]
        else:
            print(f"  ✗ Task {args.task} not found")
            return 1
    elif not args.all:
        print("  Use --all or --task N")
        return 0

    results = [run_task(t, verbose=args.verbose) for t in tasks]
    passed = sum(1 for r in results if r["match"])
    total = len(results)
    print(f"\n  Results: {passed}/{total} matched expected")

    report = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "total_tasks": total, "passed": passed,
        "failed": total - passed, "results": results,
    }
    output_path = args.output or (
        f"eval_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {output_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
