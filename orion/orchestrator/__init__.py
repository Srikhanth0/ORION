"""ORION orchestrator — AgentScope pipeline wrappers.

Exports
-------
- ``OrionModelWrapper`` — bridges AdaptiveLLMRouter ↔ AgentScope
- ``OrionPipeline`` — HiClaw pipeline factory
- ``TaskDispatcher`` — task routing with timeout enforcement
- ``execute_dag`` — DAG-based parallel subtask execution
"""
from __future__ import annotations

from orion.orchestrator.dispatcher import TaskDispatcher, execute_dag
from orion.orchestrator.model_wrapper import OrionModelWrapper
from orion.orchestrator.pipeline import OrionPipeline

__all__: list[str] = [
    "OrionModelWrapper",
    "OrionPipeline",
    "TaskDispatcher",
    "execute_dag",
]
