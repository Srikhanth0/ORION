"""Integration tests for Composio MCP tool layer.

Uses fixtures, no live API calls. Tests the full invocation pipeline:
registry → MCPClient → safety checks → result parsing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    ComposioToolWrapper,
    ToolCategory,
    ToolRegistry,
)
from orion.tools.selector import ToolSelector

_FIXTURES = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "mock_tool_responses.json"
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset the singleton before each test."""
    ToolRegistry.reset()


@pytest.fixture()
def mock_responses() -> dict[str, Any]:
    """Load mock tool responses fixture."""
    with open(_FIXTURES) as f:
        return json.load(f)


@pytest.fixture()
def registry() -> ToolRegistry:
    """Create a registry with test tools."""
    reg = ToolRegistry()
    reg.register(ComposioToolWrapper(
        name="GITHUB_CREATE_ISSUE",
        description="Create a GitHub issue",
        category=ToolCategory.GITHUB,
    ))
    reg.register(ComposioToolWrapper(
        name="FILE_WRITE",
        description="Write content to a file",
        category=ToolCategory.OS,
        is_destructive=True,
    ))
    reg.register(ComposioToolWrapper(
        name="SHELL_EXEC_CMD",
        description="Execute a shell command",
        category=ToolCategory.OS,
        is_destructive=True,
    ))
    reg.register(ComposioToolWrapper(
        name="BROWSER_NAVIGATE_URL",
        description="Navigate to a URL",
        category=ToolCategory.BROWSER,
    ))
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


class TestMCPIntegration:
    """Integration tests for the MCPClient pipeline."""

    @staticmethod
    async def _mock_composio_call(
        tool_name: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """Mock Composio call returning simulated data."""
        return {
            "status": "simulated",
            "tool": tool_name,
            "params": params,
        }

    @pytest.mark.asyncio
    async def test_invoke_registered_tool(
        self, registry: ToolRegistry
    ) -> None:
        """Successfully invoke a registered tool (simulated)."""
        client = MCPClient(registry=registry)
        client._composio_call = self._mock_composio_call
        result = await client.invoke(
            "GITHUB_CREATE_ISSUE",
            {"repo": "test/repo", "title": "Test"},
            task_id="t_int1",
        )
        assert result.ok is True
        assert result.duration_ms >= 0

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
        registry.register(ComposioToolWrapper(
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
        rollback = RollbackEngine(
            checkpoint_dir=tmp_path / "cp"
        )
        client = MCPClient(
            registry=registry,
            rollback_engine=rollback,
        )
        client._composio_call = self._mock_composio_call

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
        registry.register(ComposioToolWrapper(
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
