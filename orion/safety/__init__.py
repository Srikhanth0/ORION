"""ORION safety layer — permission, sandbox, rollback, and gate.

Exports
-------
- ``PermissionManifest`` — YAML-driven permission checking
- ``ExecSandbox`` — subprocess execution with resource limits
- ``RollbackEngine`` — per-task checkpoint/restore
- ``DestructiveOpGate`` — approval gate for destructive operations
"""
from __future__ import annotations

from orion.safety.gate import ApprovalResult, DestructiveOpGate
from orion.safety.manifest import PermissionManifest
from orion.safety.rollback import RollbackEngine, RollbackPoint
from orion.safety.sandbox import ExecSandbox

__all__: list[str] = [
    "ApprovalResult",
    "DestructiveOpGate",
    "ExecSandbox",
    "PermissionManifest",
    "RollbackEngine",
    "RollbackPoint",
]
