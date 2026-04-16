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
import io
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Windows event loop fix (must be FIRST, before any asyncio import side-effects)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ORION-FIX: Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ORION-FIX: Add project root to sys.path so 'orion' package can be imported
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_tasks.json"


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
        from orion.agentscope_config import init_agentscope

        init_agentscope()

        from agentscope.message import Msg

        from orion.agents.executor import ExecutorAgent
        from orion.agents.planner import PlannerAgent
        from orion.agents.supervisor import SupervisorAgent
        from orion.agents.verifier import VerifierAgent

        async def _run() -> dict:
            initial = Msg(
                name="user",
                role="user",
                content=instruction,
                metadata={
                    "orion_meta": {
                        "task_id": f"eval_{task_id}",
                        "step_index": 0,
                        "retry_count": 0,
                        "rollback_available": False,
                        "trace_id": f"eval_{task_id}",
                        "context": task.get("context", {}),
                    }
                },
            )
            planner = PlannerAgent()
            plan_msg = await planner.reply(initial)
            executor = ExecutorAgent()
            exec_msg = await executor.reply(plan_msg)
            verifier = VerifierAgent()
            verify_msg = await verifier.reply(exec_msg)
            supervisor = SupervisorAgent()
            final_msg = await supervisor.reply(verify_msg)

            try:
                result_data = json.loads(final_msg.content)
                decision = result_data.get("decision", {})
            except (json.JSONDecodeError, KeyError):
                decision = {}

            return {"result": final_msg.content[:200], "decision": decision}

        # Each task gets a FRESH event loop — no state leakage between tasks
        async def run_with_fresh_loop():
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            return await _run()

        run_result = asyncio.run(run_with_fresh_loop())
        decision = run_result.get("decision", {})
        action = decision.get("action", "COMPLETE")
        status = "DONE" if action == "COMPLETE" else "FAILED"
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
        "total_tasks": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }
    output_dir = Path("eval_reports")
    output_dir.mkdir(exist_ok=True)
    output_path = args.output or (
        output_dir / f"eval_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report: {output_path}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
