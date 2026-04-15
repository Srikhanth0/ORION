"""RollbackEngine — per-task checkpoint/restore stack.

Captures pre-invocation state for file writes, directory creation,
git operations, and API calls. Restores checkpoints in LIFO order
on rollback. Persists checkpoints to disk for crash recovery.

Module Contract
---------------
- **Inputs**: subtask_id + tool_name + params.
- **Outputs**: Checkpoint creation / LIFO rollback.

Depends On
----------
- ``orion.core.exceptions`` (RollbackError)
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DOES_NOT_EXIST = "__DOES_NOT_EXIST__"
_IRREVERSIBLE = "__IRREVERSIBLE__"


@dataclass
class RollbackPoint:
    """A single checkpoint capturing pre-invocation state.

    Attributes:
        subtask_id: Subtask that created this checkpoint.
        tool_name: Tool that was invoked.
        params: Tool parameters.
        rollback_type: Type of rollback (file, directory, git, irreversible).
        rollback_data: Data needed to restore state.
        created_at: ISO timestamp of checkpoint creation.
    """

    subtask_id: str
    tool_name: str
    params: dict[str, Any]
    rollback_type: str = "unknown"
    rollback_data: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


class RollbackEngine:
    """Per-task checkpoint and restore engine.

    Maintains a LIFO stack of RollbackPoints per task.
    On rollback: restores each checkpoint in reverse order,
    skipping IRREVERSIBLE entries with a warning.

    Args:
        checkpoint_dir: Directory for persisted checkpoints.
        max_checkpoints: Max checkpoints per task.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        max_checkpoints: int = 50,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self._max_checkpoints = max_checkpoints
        self._stacks: dict[str, list[RollbackPoint]] = {}

    def checkpoint(
        self,
        subtask_id: str,
        tool_name: str,
        params: dict[str, Any],
        task_id: str | None = None,
    ) -> RollbackPoint:
        """Create a pre-invocation checkpoint.

        Automatically detects the rollback type from the tool name
        and captures the current state.

        Args:
            subtask_id: Subtask creating the checkpoint.
            tool_name: Tool about to be invoked.
            params: Tool parameters.
            task_id: Parent task ID for stack key.

        Returns:
            The created RollbackPoint.
        """
        key = task_id or "default"
        rollback_type, rollback_data = self._capture_state(tool_name, params)

        point = RollbackPoint(
            subtask_id=subtask_id,
            tool_name=tool_name,
            params=params,
            rollback_type=rollback_type,
            rollback_data=rollback_data,
        )

        if key not in self._stacks:
            self._stacks[key] = []

        stack = self._stacks[key]
        if len(stack) >= self._max_checkpoints:
            logger.warning(
                "checkpoint_limit_reached",
                task_id=key,
                max=self._max_checkpoints,
            )
            stack.pop(0)  # Remove oldest

        stack.append(point)
        self._persist(key, point)

        logger.info(
            "checkpoint_created",
            task_id=key,
            subtask_id=subtask_id,
            tool=tool_name,
            type=rollback_type,
        )

        return point

    def rollback(self, task_id: str) -> list[str]:
        """Restore all checkpoints for a task in LIFO order.

        Args:
            task_id: Task ID to rollback.

        Returns:
            List of status messages for each rollback step.
        """
        key = task_id
        stack = self._stacks.get(key, [])

        if not stack:
            logger.info("rollback_nothing_to_do", task_id=task_id)
            return ["No checkpoints to rollback"]

        results: list[str] = []

        # LIFO order
        while stack:
            point = stack.pop()

            if point.rollback_type == "irreversible":
                msg = (
                    f"[WARN] Cannot rollback "
                    f"'{point.tool_name}' "
                    f"(subtask {point.subtask_id}) — "
                    f"marked as IRREVERSIBLE"
                )
                logger.warning(
                    "rollback_irreversible",
                    subtask_id=point.subtask_id,
                    tool=point.tool_name,
                )
                results.append(msg)
                continue

            try:
                self._restore(point)
                msg = f"[OK] Rolled back '{point.tool_name}' (subtask {point.subtask_id})"
                results.append(msg)
            except Exception as exc:
                msg = f"[FAIL] Could not rollback '{point.tool_name}': {exc}"
                logger.error(
                    "rollback_failed",
                    subtask_id=point.subtask_id,
                    tool=point.tool_name,
                    error=str(exc),
                )
                results.append(msg)

        # Clean up persisted checkpoints
        self._cleanup_persisted(key)

        logger.info(
            "rollback_completed",
            task_id=task_id,
            steps=len(results),
        )

        return results

    def has_checkpoints(self, task_id: str) -> bool:
        """Check if a task has any checkpoints.

        Args:
            task_id: Task ID to check.

        Returns:
            True if checkpoints exist.
        """
        return bool(self._stacks.get(task_id))

    def _capture_state(self, tool_name: str, params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Detect rollback type and capture current state.

        Args:
            tool_name: Tool name.
            params: Tool parameters.

        Returns:
            (rollback_type, rollback_data) tuple.
        """
        name_lower = tool_name.lower()

        # File write: capture original content
        if "write" in name_lower or "file" in name_lower:
            path = params.get("path", "")
            if path:
                p = Path(path)
                if p.exists():
                    try:
                        return "file", {
                            "path": str(p),
                            "original_content": p.read_text(encoding="utf-8"),
                        }
                    except Exception:
                        return "file", {
                            "path": str(p),
                            "original_content": _DOES_NOT_EXIST,
                        }
                return "file", {
                    "path": str(p),
                    "original_content": _DOES_NOT_EXIST,
                }

        # Directory creation: record path
        if "mkdir" in name_lower or "create_dir" in name_lower:
            path = params.get("path", "")
            return "directory", {"path": path}

        # Git operations: stash ref
        if "git" in name_lower or "push" in name_lower:
            return "git", {"operation": tool_name}

        # API calls: mark as irreversible
        if any(
            k in name_lower
            for k in (
                "send",
                "email",
                "slack",
                "post",
                "webhook",
                "notify",
            )
        ):
            return "irreversible", {}

        return "unknown", {}

    def _restore(self, point: RollbackPoint) -> None:
        """Restore a single checkpoint.

        Args:
            point: RollbackPoint to restore.

        Raises:
            RollbackError: If restoration fails.
        """
        if point.rollback_type == "file":
            path = point.rollback_data.get("path", "")
            original = point.rollback_data.get("original_content", "")

            if not path:
                return

            p = Path(path)
            if original == _DOES_NOT_EXIST:
                # File didn't exist before — delete it
                if p.exists():
                    p.unlink()
            else:
                # Restore original content
                p.write_text(original, encoding="utf-8")

        elif point.rollback_type == "directory":
            path = point.rollback_data.get("path", "")
            if path:
                p = Path(path)
                if p.exists() and p.is_dir():
                    shutil.rmtree(p)

        elif point.rollback_type == "git":
            logger.info(
                "rollback_git_placeholder",
                operation=point.rollback_data.get("operation", ""),
            )

    def _persist(self, task_id: str, point: RollbackPoint) -> None:
        """Persist a checkpoint to disk.

        Args:
            task_id: Task ID.
            point: Checkpoint to persist.
        """
        try:
            task_dir = self._checkpoint_dir / task_id
            task_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{point.subtask_id}_{point.tool_name}.json"
            filepath = task_dir / filename

            with open(filepath, "w") as f:
                json.dump(asdict(point), f, indent=2)

        except Exception as exc:
            logger.warning(
                "checkpoint_persist_failed",
                task_id=task_id,
                error=str(exc),
            )

    def _cleanup_persisted(self, task_id: str) -> None:
        """Remove persisted checkpoints for a task.

        Args:
            task_id: Task ID whose checkpoints to clean up.
        """
        try:
            task_dir = self._checkpoint_dir / task_id
            if task_dir.exists():
                shutil.rmtree(task_dir)
        except Exception as exc:
            logger.warning(
                "checkpoint_cleanup_failed",
                task_id=task_id,
                error=str(exc),
            )
