"""ORION core — framework-agnostic primitives.

This subpackage defines the foundational data types used throughout
ORION. These types have ZERO dependencies on external frameworks
(AgentScope, FastAPI, etc.) — they are pure Pydantic models
and typed exceptions.

Exports
-------
- ``Task``, ``Subtask``, ``TaskDAG``, ``TaskStatus``
- ``StepResult``, ``TaskResult``, ``RollbackPoint``
- All exception classes from ``orion.core.exceptions``

Depends On
----------
- ``pydantic`` (data validation)
- Python stdlib only

Must NOT Know About
-------------------
- LLM providers or router implementation
- Agent classes or orchestrator logic
- Tool registry or MCP client
- Memory backends (ChromaDB, embeddings)
"""
from __future__ import annotations

from orion.core.exceptions import (
    AllProvidersExhaustedError,
    LLMError,
    OrionError,
    PermissionDeniedError,
    PlanError,
    QuotaExceededError,
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
    "QuotaExceededError",
    "RollbackError",
]
