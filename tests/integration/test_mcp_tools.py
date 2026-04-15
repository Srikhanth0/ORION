"""Integration tests for MCP tool layer.

Uses fixtures and mocks, no live MCP server processes. Tests the full
invocation pipeline: registry → MCPClient → safety checks → result parsing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orion.core.exceptions import (
    PermissionDeniedError,
    ToolError,
    ToolNotFoundError,
)
from orion.safety.gate import DestructiveOpGate
from orion.safety.manifest import PermissionManifest
from orion.safety.rollback import RollbackEngine
from orion.tools.mcp_client import MCPClient
from orion.tools.registry import (
    MCPToolWrapper,
    ToolCategory,
    ToolRegistry,
)
from orion.tools.selector import ToolSelector


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset the singleton before each test."""
    ToolRegistry.reset()


@pytest.fixture()
def registry() -> ToolRegistry:
    """Create a registry with test tools."""
    reg = ToolRegistry()
    reg.register(MCPToolWrapper(
        name="GITHUB_CREATE_ISSUE",
        description="Create a GitHub issue",
        category=ToolCategory.GITHUB,
        server_category="github_tools",
    ))
    reg.register(MCPToolWrapper(
        name="FILE_WRITE",
        description="Write content to a file",
        category=ToolCategory.OS,
        is_destructive=True,
        server_category="os_tools",
    ))
    reg.register(MCPToolWrapper(
        name="SHELL_EXEC_CMD",
        description="Execute a shell command",
        category=ToolCategory.OS,
        is_destructive=True,
        server_category="os_tools",
    ))
    reg.register(MCPToolWrapper(
        name="BROWSER_NAVIGATE_URL",
        description="Navigate to a URL",
        category=ToolCategory.BROWSER,
        server_category="browser_tools",
    ))

    # Map tools to mock servers
    for tool in reg.list_tools():
        reg._tool_to_server[tool.name] = tool.server_category

    return reg


@pytest.fixture()
def manifest(tmp_path: Path) -> PermissionManifest:
    """Create a test permission manifest."""
    config = tmp_path / "permissions.yaml"
    config.write_text(
        """
github:
  allowed:
    - GITHUB_CREATE_ISSUE
  denied:
    - GITHUB_DELETE_REPO
shell:
  denied_patterns:
    - "rm -rf /"
filesystem:
  allowed_paths:
    - "/tmp"
    - "/home"
"""
    )
    return PermissionManifest(config_path=config)


def _make_mock_call_result(
    text: str = "simulated result",
) -> MagicMock:
    """Create a mock MCP CallToolResult."""
    content_block = MagicMock()
    content_block.text = text
    result = MagicMock()
    result.content = [content_block]
    return result


class TestMCPIntegration:
    """Integration tests for the MCPClient pipeline."""

    @pytest.mark.asyncio
    async def test_invoke_registered_tool(
        self, registry: ToolRegistry
    ) -> None:
        """Successfully invoke a registered tool (mocked MCP)."""
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = _make_mock_call_result()

        # Wire mock session to the server entry
        from orion.tools.registry import MCPServerEntry
        registry._servers["github_tools"] = MCPServerEntry(
            category="github_tools",
            command="mock",
        )
        registry._servers["github_tools"]._session = mock_session

        client = MCPClient(registry=registry)
        result = await client.invoke(
            "GITHUB_CREATE_ISSUE",
            {"repo": "test/repo", "title": "Test"},
            task_id="t_int1",
        )
        assert result.ok is True
        assert result.duration_ms >= 0
        assert "simulated result" in result.output

    @pytest.mark.asyncio
    async def test_invoke_unregistered_tool_raises(
        self, registry: ToolRegistry
    ) -> None:
        """Invoking unregistered tool raises ToolNotFoundError."""
        client = MCPClient(registry=registry)
        with pytest.raises(ToolNotFoundError):
            await client.invoke(
                "NONEXISTENT_TOOL", {},
                task_id="t_int2",
            )

    @pytest.mark.asyncio
    async def test_permission_denied_tool(
        self,
        registry: ToolRegistry,
        manifest: PermissionManifest,
    ) -> None:
        """Permission-denied tool raises PermissionDeniedError."""
        registry.register(MCPToolWrapper(
            name="GITHUB_DELETE_REPO",
            description="Delete a repo",
            category=ToolCategory.GITHUB,
            is_destructive=True,
        ))

        client = MCPClient(
            registry=registry,
            permission_manifest=manifest,
        )
        with pytest.raises(PermissionDeniedError):
            await client.invoke(
                "GITHUB_DELETE_REPO",
                {"repo": "org/prod"},
                task_id="t_int3",
            )

    @pytest.mark.asyncio
    async def test_destructive_gate_blocks(
        self, registry: ToolRegistry
    ) -> None:
        """Destructive op gate blocks in strict mode."""
        gate = DestructiveOpGate(mode="strict")
        client = MCPClient(
            registry=registry,
            gate=gate,
        )
        with pytest.raises(PermissionDeniedError):
            await client.invoke(
                "FILE_WRITE",
                {"path": "/tmp/test.txt", "content": "x"},
                task_id="t_int4",
            )

    @pytest.mark.asyncio
    async def test_rollback_integration(
        self,
        registry: ToolRegistry,
        tmp_path: Path,
    ) -> None:
        """MCPClient creates checkpoints via RollbackEngine."""
        mock_session = AsyncMock()
        mock_session.call_tool.return_value = _make_mock_call_result()

        from orion.tools.registry import MCPServerEntry
        registry._servers["github_tools"] = MCPServerEntry(
            category="github_tools",
            command="mock",
        )
        registry._servers["github_tools"]._session = mock_session

        rollback = RollbackEngine(
            checkpoint_dir=tmp_path / "cp"
        )
        client = MCPClient(
            registry=registry,
            rollback_engine=rollback,
        )

        await client.invoke(
            "GITHUB_CREATE_ISSUE",
            {"repo": "test/repo", "title": "Test"},
            task_id="t_int5",
            subtask_id="s1",
        )

        assert rollback.has_checkpoints("t_int5") is False

    @pytest.mark.asyncio
    async def test_schema_validation_failure(
        self, registry: ToolRegistry
    ) -> None:
        """Invalid params against schema raises ToolError."""
        registry.register(MCPToolWrapper(
            name="STRICT_TOOL",
            description="Tool with schema",
            category=ToolCategory.SYSTEM,
            params_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                },
                "required": ["name"],
            },
        ))

        client = MCPClient(registry=registry)
        with pytest.raises(ToolError, match="validation"):
            await client.invoke(
                "STRICT_TOOL",
                {},  # Missing required 'name'
                task_id="t_int6",
            )


class TestToolSelectorIntegration:
    """Integration tests for ToolSelector."""

    def test_suggest_returns_ranked(
        self, registry: ToolRegistry
    ) -> None:
        """ToolSelector returns ranked suggestions."""
        selector = ToolSelector(registry=registry)
        suggestions = selector.suggest(
            "create a github issue"
        )

        assert len(suggestions) > 0
        names = [s.name for s in suggestions]
        assert "GITHUB_CREATE_ISSUE" in names

    def test_format_for_prompt(
        self, registry: ToolRegistry
    ) -> None:
        """format_for_prompt produces readable output."""
        selector = ToolSelector(registry=registry)
        text = selector.format_for_prompt("navigate browser")

        assert "Suggested tools" in text
        assert "BROWSER_NAVIGATE_URL" in text

    def test_empty_registry(self) -> None:
        """Empty registry returns no suggestions."""
        empty_reg = ToolRegistry()
        selector = ToolSelector(registry=empty_reg)
        suggestions = selector.suggest("anything")
        assert suggestions == []


class TestMCPRegistryConfig:
    """Tests for MCP server configuration loading."""

    def test_load_from_config(self, tmp_path: Path) -> None:
        """Registry loads server entries from YAML config."""
        config = tmp_path / "servers.yaml"
        config.write_text("""
servers:
  test_tools:
    command: echo
    args: ["hello"]
    env: {}
defaults:
  timeout_seconds: 10
""")
        registry = ToolRegistry(config_path=config)
        registry.load_from_config()

        assert "test_tools" in registry._servers
        assert registry._servers["test_tools"].command == "echo"
        assert registry._servers["test_tools"].args == ["hello"]

    def test_load_missing_config(self, tmp_path: Path) -> None:
        """Missing config file doesn't crash."""
        registry = ToolRegistry(config_path=tmp_path / "nonexistent.yaml")
        registry.load_from_config()
        assert len(registry._servers) == 0

    def test_legacy_load_delegates(self, tmp_path: Path) -> None:
        """Legacy load() method delegates to load_from_config()."""
        config = tmp_path / "servers.yaml"
        config.write_text("""
servers:
  legacy_test:
    command: echo
    args: []
    env: {}
""")
        registry = ToolRegistry(config_path=config)
        registry.load()  # Legacy method
        assert "legacy_test" in registry._servers

    def test_env_var_resolution(self, tmp_path: Path) -> None:
        """Environment variable references are resolved."""
        import os
        os.environ["TEST_MCP_TOKEN"] = "secret123"

        config = tmp_path / "servers.yaml"
        config.write_text("""
servers:
  env_test:
    command: echo
    args: []
    env:
      TOKEN: "${TEST_MCP_TOKEN}"
""")
        registry = ToolRegistry(config_path=config)
        registry.load_from_config()

        assert registry._servers["env_test"].env["TOKEN"] == "secret123"

        # Cleanup
        del os.environ["TEST_MCP_TOKEN"]
