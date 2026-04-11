"""ORION — Hierarchical OS Automation Agent.

Combines AgentScope orchestration, HiClaw agent clusters,
Composio MCP tool integrations, and a tiered LLM backend
(Qwen 2.5 / vLLM → Groq free tier → OpenRouter).

Public API Surface
------------------
- ``Task``, ``Subtask``, ``TaskDAG`` — planning primitives
- ``StepResult``, ``TaskResult`` — execution result containers
- ``TaskStatus`` — lifecycle enum

All other symbols are internal. Import subpackages directly
for lower-level access (e.g., ``from orion.llm import ...``).
"""
from __future__ import annotations

__version__: str = "0.1.0"

# ── Public re-exports from core ──────────────────────────────
from orion.core.exceptions import (
    AllProvidersExhaustedError,
    LLMError,
    OrionError,
    PermissionDeniedError,
    PlanError,
    RollbackError,
    SafetyError,
    SandboxViolationError,
    TaskValidationError,
    ToolError,
    ToolNotFoundError,
    ToolTimeoutError,
)
from orion.core.result import RollbackPoint, StepResult, TaskResult
from orion.core.task import Subtask, Task, TaskDAG, TaskStatus

__all__: list[str] = [
    "__version__",
    # Task primitives
    "Task",
    "Subtask",
    "TaskDAG",
    "TaskStatus",
    # Result containers
    "StepResult",
    "TaskResult",
    "RollbackPoint",
    # Exceptions
    "OrionError",
    "TaskValidationError",
    "PlanError",
    "ToolError",
    "ToolNotFoundError",
    "ToolTimeoutError",
    "SafetyError",
    "PermissionDeniedError",
    "SandboxViolationError",
    "LLMError",
    "AllProvidersExhaustedError",
    "RollbackError",
]
