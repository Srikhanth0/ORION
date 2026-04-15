"""Unit tests for ToolRegistry."""

from __future__ import annotations

import pytest

from orion.core.exceptions import ToolNotFoundError
from orion.tools.registry import (
    ComposioToolWrapper,
    MCPToolWrapper,
    ToolCategory,
    ToolRegistry,
)


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    """Reset the singleton before each test."""
    ToolRegistry.reset()


def _make_tool(
    name: str = "GITHUB_CREATE_ISSUE",
    description: str = "Create a GitHub issue",
    category: ToolCategory = ToolCategory.GITHUB,
    is_destructive: bool = False,
) -> MCPToolWrapper:
    """Create a test tool wrapper."""
    return MCPToolWrapper(
        name=name,
        description=description,
        category=category,
        is_destructive=is_destructive,
    )


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self) -> None:
        """Register a tool and retrieve it by name."""
        registry = ToolRegistry()
        tool = _make_tool()
        registry.register(tool)

        result = registry.get("GITHUB_CREATE_ISSUE")
        assert result.name == "GITHUB_CREATE_ISSUE"
        assert result.category == ToolCategory.GITHUB

    def test_get_not_found_raises(self) -> None:
        """Getting an unregistered tool raises ToolNotFoundError."""
        registry = ToolRegistry()

        with pytest.raises(ToolNotFoundError, match="NONEXISTENT"):
            registry.get("NONEXISTENT")

    def test_not_found_error_has_available_tools(self) -> None:
        """ToolNotFoundError includes list of available tools."""
        registry = ToolRegistry()
        registry.register(_make_tool("TOOL_A"))

        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get("TOOL_B")

        assert "TOOL_A" in exc_info.value.available_tools

    def test_describe_all(self) -> None:
        """describe_all generates compact tool descriptions."""
        registry = ToolRegistry()
        registry.register(_make_tool("TOOL_A", "Does A", is_destructive=True))
        registry.register(_make_tool("TOOL_B", "Does B", is_destructive=False))

        desc = registry.describe_all()
        assert "TOOL_A" in desc
        assert "TOOL_B" in desc
        assert "destructive=T" in desc
        assert "destructive=F" in desc

    def test_tool_count(self) -> None:
        """tool_count returns number of registered tools."""
        registry = ToolRegistry()
        assert registry.tool_count == 0

        registry.register(_make_tool("T1"))
        registry.register(_make_tool("T2"))
        assert registry.tool_count == 2

    def test_list_tools(self) -> None:
        """list_tools returns all registered wrappers."""
        registry = ToolRegistry()
        registry.register(_make_tool("T1"))
        registry.register(_make_tool("T2"))

        tools = registry.list_tools()
        names = {t.name for t in tools}
        assert names == {"T1", "T2"}

    def test_keyword_score(self) -> None:
        """Keyword-based scoring returns relevant tools."""
        registry = ToolRegistry()
        registry.register(_make_tool("FILE_WRITE", "Write content to a file"))
        registry.register(_make_tool("GITHUB_CREATE_ISSUE", "Create a GitHub issue"))

        scored = registry.score("write a file", top_k=2)
        assert len(scored) > 0
        assert scored[0].tool.name == "FILE_WRITE"

    def test_singleton_pattern(self) -> None:
        """get_instance returns the same singleton."""
        a = ToolRegistry.get_instance()
        b = ToolRegistry.get_instance()
        assert a is b

    def test_destructive_detection(self) -> None:
        """Tools with destructive keywords are flagged."""
        registry = ToolRegistry()
        tool = registry._wrap_tool(
            type(
                "T",
                (),
                {
                    "name": "GITHUB_DELETE_REPO",
                    "description": "Delete a repo",
                    "parameters": {},
                },
            )()
        )
        assert tool.is_destructive is True

    def test_category_inference(self) -> None:
        """Category is inferred from tool name prefix."""
        registry = ToolRegistry()
        assert registry._infer_category("GITHUB_X") == ToolCategory.GITHUB
        assert registry._infer_category("SHELL_X") == ToolCategory.OS
        assert registry._infer_category("BROWSER_X") == ToolCategory.BROWSER
        assert registry._infer_category("SLACK_X") == ToolCategory.SAAS
        assert registry._infer_category("UNKNOWN") == ToolCategory.SYSTEM

    def test_backward_compat_alias(self) -> None:
        """ComposioToolWrapper is an alias for MCPToolWrapper."""
        assert ComposioToolWrapper is MCPToolWrapper
        tool = ComposioToolWrapper(
            name="COMPAT_TOOL",
            description="Backward compat test",
        )
        assert isinstance(tool, MCPToolWrapper)

    def test_mcp_tool_wrapper_server_category(self) -> None:
        """MCPToolWrapper has server_category field."""
        tool = MCPToolWrapper(
            name="read_file",
            description="Read a file",
            server_category="os_tools",
        )
        assert tool.server_category == "os_tools"
