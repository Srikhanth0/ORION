"""Task planning primitives for ORION.

Defines the core data structures used by the Planner agent to
decompose a user instruction into an executable directed acyclic
graph (DAG) of subtasks.

Module Contract
---------------
- **Inputs**: Raw user instruction (str), optional metadata dict.
- **Outputs**: ``Task``, ``Subtask``, ``TaskDAG`` frozen Pydantic models.
- **Failure Modes**: ``TaskValidationError`` on invalid DAG topology
  (cycles, missing dependency references, empty subtask lists).
- **Must NOT Know About**: LLM providers, tool registry, agent
  implementations, memory backends, or orchestrator logic.

Depends On
----------
- ``pydantic`` for data validation.
- ``orion.core.exceptions.TaskValidationError`` for error signalling.
"""

from __future__ import annotations

import enum
from collections import deque
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from orion.core.exceptions import TaskValidationError


class TaskStatus(enum.StrEnum):
    """Lifecycle status of a task.

    States follow a linear progression with possible terminal
    branches to FAILED or ROLLED_BACK.

    ::

        PENDING → PLANNING → EXECUTING → VERIFYING → COMPLETED
                      ↘          ↘           ↘
                     FAILED     FAILED     FAILED
                                  ↘
                             ROLLED_BACK
    """

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class Subtask(BaseModel, frozen=True):
    """A single atomic unit of work within a TaskDAG.

    Each subtask maps to exactly one tool invocation. The Executor
    agent processes subtasks in topological order, respecting the
    ``depends_on`` edges.

    Args:
        id: Unique identifier for this subtask (auto-generated if omitted).
        action: Human-readable description of what this subtask does.
        tool_hint: Suggested tool name from the registry (Planner's best guess).
        depends_on: List of subtask IDs that must complete before this one.
        params: Key-value parameters to pass to the tool.
        timeout_seconds: Maximum execution time for this subtask.
    """

    id: str = Field(default_factory=lambda: f"s_{uuid4().hex[:8]}")
    action: str
    tool_hint: str = ""
    depends_on: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 30


class TaskDAG(BaseModel, frozen=True):
    """A directed acyclic graph of subtasks representing an execution plan.

    The Planner agent produces a ``TaskDAG`` from a user instruction.
    The Executor agent consumes it by iterating ``topological_order()``.

    Validation happens eagerly: constructing a ``TaskDAG`` with cyclic
    or structurally invalid dependencies will raise ``TaskValidationError``.

    Args:
        task_id: Unique identifier for the parent task.
        instruction: The original user instruction that produced this DAG.
        subtasks: Ordered list of subtasks. Execution order is determined
            by ``topological_order()``, not by list position.

    Raises:
        TaskValidationError: If the DAG contains cycles, references
            non-existent subtask IDs, or has an empty subtask list.
    """

    task_id: str = Field(default_factory=lambda: f"t_{uuid4().hex[:8]}")
    instruction: str
    subtasks: list[Subtask]

    def model_post_init(self, __context: Any) -> None:
        """Validate DAG structure after construction."""
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Check that the DAG is well-formed.

        Raises:
            TaskValidationError: On empty subtasks, missing deps, or cycles.
        """
        if not self.subtasks:
            raise TaskValidationError(
                "TaskDAG must contain at least one subtask.",
                task_id=self.task_id,
            )

        known_ids = {s.id for s in self.subtasks}

        # Check for duplicate IDs
        if len(known_ids) != len(self.subtasks):
            seen: set[str] = set()
            for s in self.subtasks:
                if s.id in seen:
                    raise TaskValidationError(
                        f"Duplicate subtask ID: '{s.id}'.",
                        task_id=self.task_id,
                        subtask_id=s.id,
                    )
                seen.add(s.id)

        # Check for references to non-existent subtask IDs
        for subtask in self.subtasks:
            for dep_id in subtask.depends_on:
                if dep_id not in known_ids:
                    raise TaskValidationError(
                        f"Subtask '{subtask.id}' depends on unknown subtask '{dep_id}'.",
                        task_id=self.task_id,
                        subtask_id=subtask.id,
                    )

        # Cycle detection via Kahn's algorithm (dry run)
        self._detect_cycles(known_ids)

    def _detect_cycles(self, known_ids: set[str]) -> None:
        """Detect cycles using Kahn's algorithm.

        Args:
            known_ids: Set of all subtask IDs in this DAG.

        Raises:
            TaskValidationError: If a cycle is detected, with the
                cycle_path field populated.
        """
        # Build adjacency list and in-degree map
        in_degree: dict[str, int] = dict.fromkeys(known_ids, 0)
        adjacency: dict[str, list[str]] = {sid: [] for sid in known_ids}

        for subtask in self.subtasks:
            for dep_id in subtask.depends_on:
                adjacency[dep_id].append(subtask.id)
                in_degree[subtask.id] += 1

        # Kahn's: start with zero-in-degree nodes
        queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(known_ids):
            # Nodes still with in_degree > 0 form the cycle
            cycle_nodes = [sid for sid, deg in in_degree.items() if deg > 0]
            raise TaskValidationError(
                f"TaskDAG contains a dependency cycle involving: {cycle_nodes}.",
                task_id=self.task_id,
                cycle_path=cycle_nodes,
            )

    def topological_order(self) -> list[Subtask]:
        """Return subtasks in a valid execution order.

        Uses Kahn's algorithm to produce a topological sort. Since
        structure is validated at construction time, this method is
        guaranteed to succeed.

        Returns:
            List of ``Subtask`` instances in dependency-respecting order.
        """
        id_to_subtask = {s.id: s for s in self.subtasks}
        in_degree: dict[str, int] = {s.id: 0 for s in self.subtasks}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.subtasks}

        for subtask in self.subtasks:
            for dep_id in subtask.depends_on:
                adjacency[dep_id].append(subtask.id)
                in_degree[subtask.id] += 1

        queue: deque[str] = deque(sid for sid, deg in in_degree.items() if deg == 0)
        result: list[Subtask] = []

        while queue:
            node = queue.popleft()
            result.append(id_to_subtask[node])
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def get_subtask(self, subtask_id: str) -> Subtask:
        """Look up a subtask by ID.

        Args:
            subtask_id: The ID to search for.

        Returns:
            The matching ``Subtask`` instance.

        Raises:
            TaskValidationError: If the ID is not found in this DAG.
        """
        for subtask in self.subtasks:
            if subtask.id == subtask_id:
                return subtask
        raise TaskValidationError(
            f"Subtask '{subtask_id}' not found in TaskDAG '{self.task_id}'.",
            task_id=self.task_id,
            subtask_id=subtask_id,
        )


class Task(BaseModel):
    """Top-level task object representing a user request lifecycle.

    Unlike ``TaskDAG`` (frozen, immutable once planned), ``Task`` is
    mutable — its ``status`` field updates as the task progresses
    through the pipeline.

    Args:
        task_id: Unique identifier (auto-generated if omitted).
        instruction: The original user instruction.
        created_at: UTC timestamp of task creation.
        dag: The execution plan, set by the Planner agent.
        status: Current lifecycle status.
        metadata: Arbitrary key-value metadata (user prefs, OS context).
    """

    task_id: str = Field(default_factory=lambda: f"t_{uuid4().hex[:8]}")
    instruction: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dag: TaskDAG | None = None
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] = Field(default_factory=dict)
