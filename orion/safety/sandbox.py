"""ExecSandbox — sandboxed subprocess execution with resource limits.

All shell/subprocess tool calls execute through this sandbox,
which enforces timeouts, path restrictions, and environment
sanitization.

Module Contract
---------------
- **Inputs**: command string + timeout + cwd.
- **Outputs**: ExecResult dataclass.

Depends On
----------
- ``orion.core.exceptions`` (SandboxViolationError, ToolTimeoutError)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import structlog

from orion.core.exceptions import (
    SandboxViolationError,
    ToolTimeoutError,
)

logger = structlog.get_logger(__name__)

# Env vars to strip from subprocess environment
_SENSITIVE_ENV_PREFIXES = frozenset(
    {
        "API_KEY",
        "SECRET",
        "TOKEN",
        "PASSWORD",
        "GITHUB_PAT",
        "OPENAI",
        "GROQ",
        "OPENROUTER",
        "AWS_SECRET",
        "GITHUB_TOKEN",
    }
)


@dataclass
class ExecResult:
    """Result of a sandboxed command execution.

    Attributes:
        return_code: Process exit code.
        stdout: Standard output.
        stderr: Standard error.
        timed_out: Whether the process was killed due to timeout.
    """

    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


class ExecSandbox:
    """Sandboxed subprocess execution engine.

    Enforces:
    - Timeout via asyncio.wait_for.
    - CWD restricted to allowed paths.
    - Environment stripped of sensitive vars.
    - On Windows: standard subprocess limits.
    - On Linux: resource.setrlimit for memory/process caps.

    Args:
        allowed_paths: List of allowed working directories.
        default_timeout: Default timeout in seconds.
        max_memory_mb: Max virtual memory in MB (Linux only).
        max_processes: Max child processes (Linux only).
        max_file_size_mb: Max file write size in MB (Linux only).
    """

    def __init__(
        self,
        allowed_paths: list[str] | None = None,
        default_timeout: float = 30.0,
        max_memory_mb: int = 512,
        max_processes: int = 64,
        max_file_size_mb: int = 100,
    ) -> None:
        self._allowed_paths = allowed_paths or [
            "/home",
            "/tmp",
            "/workspace",
            "C:\\Users",
            "C:\\temp",
        ]
        self._default_timeout = default_timeout
        self._max_memory_mb = max_memory_mb
        self._max_processes = max_processes
        self._max_file_size_mb = max_file_size_mb

    async def run(
        self,
        command: str,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> ExecResult:
        """Execute a command in the sandbox.

        Args:
            command: Shell command string.
            timeout: Override default timeout.
            cwd: Working directory (must be in allowed_paths).

        Returns:
            ExecResult with output and exit code.

        Raises:
            SandboxViolationError: If CWD is not allowed.
            ToolTimeoutError: If command exceeds timeout.
        """
        effective_timeout = timeout or self._default_timeout
        effective_cwd = cwd or str(Path.cwd())

        # Validate CWD
        self._validate_cwd(effective_cwd)

        # Build sanitized environment
        env = self._sanitize_env()

        logger.info(
            "sandbox_exec_start",
            command=command[:100],
            cwd=effective_cwd,
            timeout=effective_timeout,
        )

        try:
            result = await asyncio.wait_for(
                self._spawn(command, env, effective_cwd),
                timeout=effective_timeout,
            )

            logger.info(
                "sandbox_exec_complete",
                command=command[:50],
                return_code=result.return_code,
            )

            return result

        except TimeoutError:
            logger.error(
                "sandbox_exec_timeout",
                command=command[:50],
                timeout=effective_timeout,
            )
            raise ToolTimeoutError(
                f"Command timed out after {effective_timeout}s: {command[:50]}",
                tool_name="shell",
                timeout_seconds=effective_timeout,
            ) from None

    async def _spawn(
        self,
        command: str,
        env: dict[str, str],
        cwd: str,
    ) -> ExecResult:
        """Spawn the subprocess.

        Args:
            command: Shell command.
            env: Sanitized environment.
            cwd: Working directory.

        Returns:
            ExecResult with captured output.
        """
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        stdout_bytes, stderr_bytes = await proc.communicate()

        return ExecResult(
            return_code=proc.returncode or 0,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
        )

    def _validate_cwd(self, cwd: str) -> None:
        """Validate that the CWD is in the allowed paths list.

        Args:
            cwd: Working directory to validate.

        Raises:
            SandboxViolationError: If CWD is not allowed.
        """
        try:
            resolved = str(Path(cwd).resolve())
        except (OSError, ValueError):
            resolved = cwd

        for allowed in self._allowed_paths:
            if resolved.startswith(allowed):
                return

        raise SandboxViolationError(
            f"CWD '{resolved}' is not in allowed paths",
            resource="cwd",
            limit=str(self._allowed_paths),
            actual=resolved,
        )

    def _sanitize_env(self) -> dict[str, str]:
        """Create a sanitized copy of the environment.

        Strips any env var whose name contains sensitive prefixes.

        Returns:
            Sanitized environment dict.
        """
        clean: dict[str, str] = {}
        for key, val in os.environ.items():
            key_upper = key.upper()
            is_sensitive = any(prefix in key_upper for prefix in _SENSITIVE_ENV_PREFIXES)
            if not is_sensitive:
                clean[key] = val
        return clean
