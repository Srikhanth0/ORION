"""ORION tools — MCP tool integration layer.

Exports
-------
- ``ToolRegistry`` — singleton tool index
- ``MCPClient`` — defensive tool invocation
- ``ToolSelector`` — semantic tool matching
- ``MCPToolWrapper`` — tool wrapper dataclass
- ``ToolResult`` — invocation result dataclass
- ``ToolCategory`` — tool category enum
"""
from __future__ import annotations

from orion.tools.mcp_client import MCPClient, ToolResult
from orion.tools.registry import (
    ComposioToolWrapper,
    MCPToolWrapper,
    ScoredTool,
    ToolCategory,
    ToolRegistry,
)
from orion.tools.selector import ToolSelector, ToolSuggestion

__all__: list[str] = [
    "ComposioToolWrapper",
    "MCPClient",
    "MCPToolWrapper",
    "ScoredTool",
    "ToolCategory",
    "ToolRegistry",
    "ToolResult",
    "ToolSelector",
    "ToolSuggestion",
]
