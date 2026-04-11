"""ORION exception hierarchy.

Every exception carries structured context so that error handlers,
the Supervisor agent, and observability tooling can make informed
decisions without parsing message strings.

Hierarchy
---------
::

    OrionError
    ├── TaskValidationError
    ├── PlanError
    ├── ToolError
    │   ├── ToolNotFoundError
    │   └── ToolTimeoutError
    ├── SafetyError
    │   ├── PermissionDeniedError
    │   └── SandboxViolationError
    ├── LLMError
    │   ├── AllProvidersExhaustedError
    │   └── QuotaExceededError
    └── RollbackError

Depends On
----------
- Python stdlib only.

Must NOT Know About
-------------------
- Any ORION module. These are pure value types.
"""
from __future__ import annotations


class OrionError(Exception):
    """Base exception for all ORION errors.

    Args:
        message: Human-readable error description.
        task_id: ID of the task that triggered the error, if applicable.
        subtask_id: ID of the subtask that triggered the error, if applicable.
        context: Arbitrary structured context for observability.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.context = context or {}


# ── Task Validation ──────────────────────────────────────────


class TaskValidationError(OrionError):
    """Raised when a TaskDAG is structurally invalid.

    Examples: cyclic dependencies, references to non-existent subtask IDs,
    or empty subtask lists.

    Args:
        message: Description of the validation failure.
        task_id: ID of the invalid task.
        cycle_path: If a cycle was detected, the list of subtask IDs forming the cycle.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        cycle_path: list[str] | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.cycle_path = cycle_path


# ── Planning ─────────────────────────────────────────────────


class PlanError(OrionError):
    """Raised when the Planner agent fails to produce a valid execution plan.

    Args:
        message: Description of the planning failure.
        raw_output: The raw LLM output that could not be parsed, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.raw_output = raw_output


# ── Tool Errors ──────────────────────────────────────────────


class ToolError(OrionError):
    """Raised when a Composio/MCP tool invocation fails.

    Args:
        message: Description of the tool failure.
        tool_name: The name of the tool that failed.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        tool_name: str | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Raised when a requested tool name is not in the registry.

    Args:
        message: Description of the lookup failure.
        tool_name: The tool name that was not found.
        available_tools: List of tool names that ARE registered, for diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        tool_name: str | None = None,
        available_tools: list[str] | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            tool_name=tool_name,
        )
        self.available_tools = available_tools or []


class ToolTimeoutError(ToolError):
    """Raised when a tool invocation exceeds its configured timeout.

    Args:
        message: Description of the timeout.
        tool_name: The tool that timed out.
        timeout_seconds: The configured timeout that was exceeded.
        elapsed_seconds: How long the tool actually ran.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        tool_name: str | None = None,
        timeout_seconds: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            tool_name=tool_name,
        )
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


# ── Safety Errors ────────────────────────────────────────────


class SafetyError(OrionError):
    """Raised when a safety constraint is violated.

    Args:
        message: Description of the safety violation.
        rule: The safety rule or manifest entry that was violated.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        rule: str | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.rule = rule


class PermissionDeniedError(SafetyError):
    """Raised when a tool/action is blocked by the permission manifest.

    Args:
        message: Description of the denial.
        tool_name: The tool that was denied.
        action: The specific action attempted.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        rule: str | None = None,
        tool_name: str | None = None,
        action: str | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            rule=rule,
        )
        self.tool_name = tool_name
        self.action = action


class SandboxViolationError(SafetyError):
    """Raised when an execution exceeds sandbox resource limits.

    Args:
        message: Description of the violation.
        resource: The resource that was exceeded (cpu, memory, time, path).
        limit: The configured limit.
        actual: The actual value that exceeded the limit.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        rule: str | None = None,
        resource: str | None = None,
        limit: str | None = None,
        actual: str | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            rule=rule,
        )
        self.resource = resource
        self.limit = limit
        self.actual = actual


# ── LLM Errors ───────────────────────────────────────────────


class LLMError(OrionError):
    """Raised when an LLM provider call fails.

    Args:
        message: Description of the LLM failure.
        provider: The provider name that failed (vllm, groq, openrouter).
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.provider = provider


class AllProvidersExhaustedError(LLMError):
    """Raised when every provider in the fallback chain has failed.

    Args:
        message: Description of the exhaustion.
        attempted_providers: Ordered list of providers that were tried.
        errors: Map of provider name → error message for each failure.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        provider: str | None = None,
        attempted_providers: list[str] | None = None,
        errors: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            provider=provider,
        )
        self.attempted_providers = attempted_providers or []
        self.errors = errors or {}


class QuotaExceededError(LLMError):
    """Raised when a provider's rate limit or token budget is exhausted.

    Args:
        message: Description of the quota exceedance.
        provider: The provider whose quota was exceeded.
        remaining_requests: Remaining requests if known from headers.
        remaining_tokens: Remaining tokens if known from headers.
        retry_after_seconds: Seconds to wait before retrying, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        provider: str | None = None,
        remaining_requests: int | None = None,
        remaining_tokens: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(
            message,
            task_id=task_id,
            subtask_id=subtask_id,
            context=context,
            provider=provider,
        )
        self.remaining_requests = remaining_requests
        self.remaining_tokens = remaining_tokens
        self.retry_after_seconds = retry_after_seconds


# ── Rollback Errors ──────────────────────────────────────────


class RollbackError(OrionError):
    """Raised when a checkpoint restore operation fails.

    Args:
        message: Description of the rollback failure.
        checkpoint_path: Path to the checkpoint that could not be restored.
    """

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        subtask_id: str | None = None,
        context: dict[str, object] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__(
            message, task_id=task_id, subtask_id=subtask_id, context=context
        )
        self.checkpoint_path = checkpoint_path
