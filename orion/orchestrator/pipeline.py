"""ORION pipeline factory — assembles HiClaw agent pipelines.

Creates configured AgentScope pipelines from ORION agents.
The primary pipeline is: Planner → Executor → Verifier → Supervisor.

Module Contract
---------------
- **Inputs**: Task instruction + configured agents.
- **Outputs**: ``TaskResult`` from the full pipeline run.

Depends On
----------
- ``agentscope.pipeline`` (SequentialPipeline)
- ``agentscope.message`` (Msg)
- ``orion.agents`` (PlannerAgent, ExecutorAgent, VerifierAgent, SupervisorAgent)
"""
from __future__ import annotations

from typing import Any

from agentscope.message import Msg
from agentscope.pipeline import SequentialPipeline


class OrionPipeline:
    """Factory for assembling HiClaw agent pipelines.

    Creates a SequentialPipeline: Planner → Executor → Verifier → Supervisor.
    Each agent receives the previous agent's Msg output, carrying
    structured data in ``orion_meta``.

    Args:
        planner: Configured PlannerAgent instance.
        executor: Configured ExecutorAgent instance.
        verifier: Configured VerifierAgent instance.
        supervisor: Configured SupervisorAgent instance.
    """

    def __init__(
        self,
        planner: Any,
        executor: Any,
        verifier: Any,
        supervisor: Any,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._supervisor = supervisor

        self._pipeline = SequentialPipeline(
            operators=[planner, executor, verifier, supervisor],
        )

    async def run(
        self,
        instruction: str,
        task_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Msg:
        """Execute the full HiClaw pipeline for a task.

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

        # Run the pipeline
        result = await self._pipeline(initial_msg)
        return result

    @property
    def agents(self) -> list[Any]:
        """Return the ordered list of agents in this pipeline."""
        return [
            self._planner,
            self._executor,
            self._verifier,
            self._supervisor,
        ]
