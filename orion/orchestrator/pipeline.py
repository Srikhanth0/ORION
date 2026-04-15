"""ORION pipeline factory — assembles HiClaw agent pipelines.

Creates configured AgentScope pipelines from ORION agents.
The primary pipeline is: Planner → DAG Executor → Verifier → Supervisor.

Subtask execution uses DAG-based parallelism: independent subtasks
within a plan execute concurrently via ``asyncio.gather``.

Module Contract
---------------
- **Inputs**: Task instruction + configured agents.
- **Outputs**: ``TaskResult`` from the full pipeline run.

Depends On
----------
- ``agentscope.message`` (Msg)
- ``orion.agents`` (PlannerAgent, ExecutorAgent, VerifierAgent, SupervisorAgent)
- ``orion.orchestrator.dispatcher`` (execute_dag)
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from agentscope.message import Msg

from orion.orchestrator.dispatcher import execute_dag

logger = structlog.get_logger(__name__)


class OrionPipeline:
    """Factory for assembling HiClaw agent pipelines.

    Executes a hybrid pipeline:
    1. **Planner** produces a subtask DAG (sequential).
    2. **Executor** runs subtasks with DAG parallelism via ``execute_dag``.
    3. **Verifier** checks results (sequential).
    4. **Supervisor** decides next action (sequential).

    Args:
        planner: Configured PlannerAgent instance.
        executor: Configured ExecutorAgent instance.
        verifier: Configured VerifierAgent instance.
        supervisor: Configured SupervisorAgent instance.
        use_parallel_dag: Enable DAG-based parallel execution (default True).
    """

    def __init__(
        self,
        planner: Any,
        executor: Any,
        verifier: Any,
        supervisor: Any,
        use_parallel_dag: bool = True,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._supervisor = supervisor
        self._use_parallel_dag = use_parallel_dag

    async def run(
        self,
        instruction: str,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Msg:
        """Execute the full HiClaw pipeline for a task.

        Pipeline flow:
        1. Planner generates subtask DAG.
        2. Subtasks execute via DAG dispatcher (parallel batches).
        3. Verifier checks execution results.
        4. Supervisor decides: COMPLETE, RETRY, ROLLBACK, or ESCALATE.

        Args:
            instruction: Natural language task instruction.
            task_id: Optional task ID (auto-generated if None).
            context: Optional OS context (cwd, hostname, user).

        Returns:
            Final Msg from the Supervisor with TaskResult.
        """
        from uuid import uuid4

        if task_id is None:
            task_id = f"t_{uuid4().hex[:8]}"

        # Build initial message
        initial_msg = Msg(
            name="user",
            role="user",
            content=instruction,
            metadata={
                "orion_meta": {
                    "task_id": task_id,
                    "subtask_id": None,
                    "step_index": 0,
                    "retry_count": 0,
                    "rollback_available": False,
                    "trace_id": f"trace_{uuid4().hex[:12]}",
                    "context": context or {},
                },
            },
        )

        # Step 1: Planner → subtask DAG
        logger.info("pipeline_step", step="planner", task_id=task_id)
        plan_msg = await self._planner.reply(initial_msg)

        # Step 2: Execute subtasks
        if self._use_parallel_dag:
            # DAG-based parallel execution
            executor_msg = await self._execute_with_dag(
                plan_msg=plan_msg,
                task_id=task_id,
            )
        else:
            # Sequential fallback (original behavior)
            logger.info("pipeline_step", step="executor_sequential", task_id=task_id)
            executor_msg = await self._executor.reply(plan_msg)

        # Step 3: Verifier
        logger.info("pipeline_step", step="verifier", task_id=task_id)
        verifier_msg = await self._verifier.reply(executor_msg)

        # Step 4: Supervisor
        logger.info("pipeline_step", step="supervisor", task_id=task_id)
        supervisor_msg = await self._supervisor.reply(verifier_msg)

        return supervisor_msg

    async def _execute_with_dag(
        self,
        plan_msg: Msg,
        task_id: str,
    ) -> Msg:
        """Execute subtasks using DAG-based parallel dispatcher.

        Parses the plan, runs parallel batches, and wraps results
        back into a Msg for the Verifier.

        Args:
            plan_msg: Planner's output Msg with JSON plan.
            task_id: Parent task ID.

        Returns:
            Msg with execution results array in content.
        """
        logger.info("pipeline_step", step="executor_dag", task_id=task_id)

        # Parse plan
        content = plan_msg.content if isinstance(plan_msg.content, str) else str(plan_msg.content)
        try:
            plan = json.loads(content)
        except json.JSONDecodeError:
            plan = {}

        subtasks = plan.get("subtasks", [])

        if not subtasks:
            # No subtasks — return empty results
            results_list: list[dict[str, Any]] = []
        else:
            # Execute via DAG dispatcher
            results_dict = await execute_dag(
                subtasks=subtasks,
                executor=self._executor,
                task_id=task_id,
            )

            # Preserve topological order in output
            from orion.agents.executor import topological_sort

            ordered_ids = topological_sort(subtasks)
            results_list = [results_dict[sid] for sid in ordered_ids if sid in results_dict]

        # Build executor output Msg
        meta = getattr(plan_msg, "metadata", {}) or {}
        orion_meta = dict(meta.get("orion_meta", {}))
        orion_meta["step_index"] = 2

        return Msg(
            name="executor",
            role="assistant",
            content=json.dumps(results_list, indent=2),
            metadata={"orion_meta": orion_meta},
        )

    @property
    def agents(self) -> list[Any]:
        """Return the ordered list of agents in this pipeline."""
        return [
            self._planner,
            self._executor,
            self._verifier,
            self._supervisor,
        ]
