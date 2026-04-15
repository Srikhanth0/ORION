"""MCPClient — defensive tool invocation with full safety pipeline.

Validates params → checks permissions → gates destructive ops →
checkpoints → executes via MCP registry → parses result → emits metrics.

Module Contract
---------------
- **Inputs**: tool_name + params dict.
- **Outputs**: ToolResult dataclass.

Depends On
----------
- ``orion.tools.registry`` (ToolRegistry)
- ``orion.safety.manifest`` (PermissionManifest)
- ``orion.safety.rollback`` (RollbackEngine)
- ``orion.safety.gate`` (DestructiveOpGate)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import structlog

from orion.core.exceptions import (
    PermissionDeniedError,
    ToolError,
)
from orion.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ToolResult:
    """Result of a tool invocation.

    Attributes:
        ok: Whether the invocation succeeded.
        output: Parsed output string.
        raw: Raw response from the tool.
        duration_ms: Execution time in milliseconds.
        tool_name: Name of the tool that was invoked.
    """

    ok: bool
    output: str | None
    raw: Any = None
    duration_ms: float = 0.0
    tool_name: str = ""


class MCPClient:
    """Defensive MCP tool invocation client.

    Full invocation pipeline:
    1. Validate params against JSON schema.
    2. Check PermissionManifest.
    3. If destructive: DestructiveOpGate.approve().
    4. RollbackEngine.checkpoint().
    5. Execute via ToolRegistry.call() (MCP stdio).
    6. Parse into ToolResult.
    7. Log with structlog.

    Args:
        registry: ToolRegistry instance.
        permission_manifest: PermissionManifest checker.
        rollback_engine: RollbackEngine for checkpointing.
        gate: DestructiveOpGate for approval.
        default_timeout: Default timeout in seconds.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        permission_manifest: Any = None,
        rollback_engine: Any = None,
        gate: Any = None,
        default_timeout: float = 30.0,
    ) -> None:
        self._registry = registry or ToolRegistry.get_instance()
        self._manifest = permission_manifest
        self._rollback = rollback_engine
        self._gate = gate
        self._default_timeout = default_timeout

    async def invoke(
        self,
        tool_name: str,
        params: dict[str, Any],
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        timeout: float | None = None,
    ) -> ToolResult:
        """Invoke a tool with full safety pipeline.

        Args:
            tool_name: Registered tool action name.
            params: Tool parameters dict.
            task_id: Parent task ID for logging.
            subtask_id: Subtask ID for logging.
            timeout: Override default timeout.

        Returns:
            ToolResult with invocation outcome.

        Raises:
            ToolNotFoundError: Tool not in registry.
            PermissionDeniedError: Blocked by manifest.
            ToolError: Invocation failure.
        """
        start = time.monotonic()

        # Step 1: Look up tool
        tool = self._registry.get(tool_name)

        # Step 2: Validate params
        self._validate_params(tool_name, params, tool.params_schema)

        # Step 3: Permission check
        if self._manifest is not None:
            self._manifest.check(
                tool_name, params, category=tool.category.value
            )

        # Step 4: Destructive gate
        if tool.is_destructive and self._gate is not None:
            approval = await self._gate.approve(
                tool_name, params,
                rollback_available=(
                    self._rollback is not None
                ),
            )
            if not approval.approved:
                raise PermissionDeniedError(
                    f"Destructive op '{tool_name}' denied: "
                    f"{approval.reason}",
                    tool_name=tool_name,
                )

        # Step 5: Checkpoint
        if self._rollback is not None:
            self._rollback.checkpoint(
                subtask_id or "unknown",
                tool_name, params,
            )

        # Step 6: Execute via MCP
        try:
            raw = await self._execute(
                tool_name, params,
                timeout=timeout or self._default_timeout,
            )
            elapsed = (time.monotonic() - start) * 1000

            result = ToolResult(
                ok=True,
                output=str(raw) if raw is not None else None,
                raw=raw,
                duration_ms=round(elapsed, 1),
                tool_name=tool_name,
            )

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            result = ToolResult(
                ok=False,
                output=None,
                raw=str(exc),
                duration_ms=round(elapsed, 1),
                tool_name=tool_name,
            )

        # Step 7: Log
        logger.info(
            "tool_invoked",
            tool=tool_name,
            category=tool.category.value,
            ok=result.ok,
            duration_ms=result.duration_ms,
            task_id=task_id,
            subtask_id=subtask_id,
        )

        return result

    def _validate_params(
        self,
        tool_name: str,
        params: dict[str, Any],
        schema: dict[str, Any],
    ) -> None:
        """Validate params against JSON schema.

        Args:
            tool_name: Tool name for error context.
            params: Parameters to validate.
            schema: JSON Schema to validate against.

        Raises:
            ToolError: If validation fails.
        """
        if not schema:
            return

        try:
            import jsonschema
            jsonschema.validate(params, schema)
        except Exception as exc:
            raise ToolError(
                f"Parameter validation failed for "
                f"'{tool_name}': {exc}",
                tool_name=tool_name,
            ) from exc

    async def _execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        timeout: float,
    ) -> Any:
        """Execute the tool call via MCP registry with retry.

        Retries up to 3 times with exponential backoff (1s, 2s, 4s).
        After 3 failures: returns failure result and logs.

        Args:
            tool_name: Tool to invoke.
            params: Parameters.
            timeout: Timeout in seconds.

        Returns:
            Raw tool response.
        """
        import asyncio

        max_retries = 3
        base_delay = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    self._registry.call(tool_name, params),
                    timeout=timeout,
                )
                return result
            except TimeoutError as exc:
                if attempt == max_retries:
                    raise ToolError(
                        f"Tool '{tool_name}' timed out "
                        f"after {timeout}s ({max_retries} attempts)",
                        tool_name=tool_name,
                    ) from exc
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "mcp_retry",
                    tool=tool_name,
                    attempt=attempt,
                    delay=delay,
                    error="timeout",
                )
                await asyncio.sleep(delay)
            except ToolError:
                raise
            except Exception as exc:
                if attempt == max_retries:
                    logger.error(
                        "mcp_server_unavailable",
                        tool=tool_name,
                        attempts=max_retries,
                        error=str(exc),
                    )
                    raise ToolError(
                        f"MCP server unavailable for '{tool_name}' "
                        f"after {max_retries} attempts: {exc}",
                        tool_name=tool_name,
                    ) from exc
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "mcp_retry",
                    tool=tool_name,
                    attempt=attempt,
                    delay=delay,
                    error=str(exc)[:100],
                )
                await asyncio.sleep(delay)
