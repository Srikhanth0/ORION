"""OS tool category — typed wrappers for Composio OS/shell actions.

Depends On
----------
- ``orion.tools.mcp_client`` (MCPClient, ToolResult)
"""
from __future__ import annotations

from orion.tools.mcp_client import MCPClient, ToolResult


class OSTools:
    """Typed wrappers for OS/shell Composio actions.

    Args:
        client: MCPClient instance for invocation.
    """

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    async def exec_cmd(
        self,
        command: str,
        cwd: str | None = None,
        timeout: float = 30.0,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Execute a shell command.

        Args:
            command: Shell command string.
            cwd: Working directory.
            timeout: Timeout in seconds.
            task_id: Parent task ID.

        Returns:
            ToolResult with stdout/stderr.
        """
        return await self._client.invoke(
            "SHELL_EXEC_CMD",
            {
                "command": command,
                "cwd": cwd or ".",
            },
            task_id=task_id,
            timeout=timeout,
        )

    async def read_file(
        self,
        path: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Read file contents.

        Args:
            path: File path to read.
            task_id: Parent task ID.

        Returns:
            ToolResult with file content.
        """
        return await self._client.invoke(
            "FILE_READ",
            {"path": path},
            task_id=task_id,
        )

    async def write_file(
        self,
        path: str,
        content: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Write content to a file.

        Args:
            path: File path to write.
            content: String content.
            task_id: Parent task ID.

        Returns:
            ToolResult with write confirmation.
        """
        return await self._client.invoke(
            "FILE_WRITE",
            {"path": path, "content": content},
            task_id=task_id,
        )

    async def list_dir(
        self,
        path: str = ".",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """List directory contents.

        Args:
            path: Directory path.
            task_id: Parent task ID.

        Returns:
            ToolResult with directory listing.
        """
        return await self._client.invoke(
            "FILE_LIST_DIR",
            {"path": path},
            task_id=task_id,
        )

    async def find_files(
        self,
        pattern: str,
        path: str = ".",
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Find files matching a pattern.

        Args:
            pattern: Glob or regex pattern.
            path: Search root directory.
            task_id: Parent task ID.

        Returns:
            ToolResult with matched file list.
        """
        return await self._client.invoke(
            "FILE_FIND",
            {"pattern": pattern, "path": path},
            task_id=task_id,
        )

    async def list_processes(
        self,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """List running processes.

        Args:
            task_id: Parent task ID.

        Returns:
            ToolResult with process list.
        """
        return await self._client.invoke(
            "OS_LIST_PROCESSES",
            {},
            task_id=task_id,
        )

    async def kill_process(
        self,
        pid: int,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Kill a process by PID.

        Args:
            pid: Process ID to kill.
            task_id: Parent task ID.

        Returns:
            ToolResult with kill confirmation.
        """
        return await self._client.invoke(
            "OS_KILL_PROCESS",
            {"pid": pid},
            task_id=task_id,
        )

    async def get_env_var(
        self,
        name: str,
        *,
        task_id: str | None = None,
    ) -> ToolResult:
        """Get an environment variable value.

        Args:
            name: Variable name.
            task_id: Parent task ID.

        Returns:
            ToolResult with variable value.
        """
        return await self._client.invoke(
            "OS_GET_ENV_VAR",
            {"name": name},
            task_id=task_id,
        )
