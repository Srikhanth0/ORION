"""ToolRegistry — singleton index of all available MCP tools.

Loads tool server definitions from configs/mcp/servers.yaml,
spawns MCP stdio server processes on demand, and provides
tool lookup, description, and semantic scoring for the Planner.

Module Contract
---------------
- **Inputs**: servers.yaml config + MCP server processes.
- **Outputs**: Tool lookup, compact descriptions, scored suggestions.

Depends On
----------
- ``mcp`` SDK (ClientSession, StdioServerParameters, stdio_client)
- ``orion.core.exceptions`` (ToolNotFoundError)
"""

from __future__ import annotations

import enum
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from orion.core.exceptions import ToolNotFoundError

logger = structlog.get_logger(__name__)


def _resolve_bin(name: str) -> str:
    """Return the correct executable name for the current platform."""
    if sys.platform == "win32":
        # On Windows, prefer the .cmd shim that lives in %APPDATA%\npm
        cmd_variant = name + ".cmd"
        if shutil.which(cmd_variant):
            return cmd_variant
    return name  # Linux / WSL2 / macOS — name is correct as-is


UVX_BIN = _resolve_bin("uvx")
NPX_BIN = _resolve_bin("npx")

_DEFAULT_CONFIG = Path("configs/mcp/servers.yaml")


class ToolCategory(enum.StrEnum):
    """Tool category classification."""

    GITHUB = "github"
    OS = "os"
    BROWSER = "browser"
    SAAS = "saas"
    SYSTEM = "system"
    VISION = "vision"


_DESTRUCTIVE_PATTERNS = frozenset(
    {
        "delete",
        "remove",
        "kill",
        "drop",
        "destroy",
        "purge",
        "truncate",
        "wipe",
        "erase",
        "uninstall",
    }
)


@dataclass(frozen=True)
class MCPToolWrapper:
    """Immutable wrapper around a single MCP tool action.

    Attributes:
        name: Tool action name (e.g. ``read_file``).
        description: Human-readable description from the tool.
        params_schema: JSON Schema dict for parameter validation.
        is_destructive: Whether this tool mutates external state.
        category: Tool category for permission routing.
        cost_estimate: Estimated token cost per average call.
        server_category: MCP server category this tool belongs to.
    """

    name: str
    description: str
    params_schema: dict[str, Any] = field(default_factory=dict)
    is_destructive: bool = False
    category: ToolCategory = ToolCategory.SYSTEM
    cost_estimate: float = 0.0
    server_category: str = ""


# Backward-compatible alias
ComposioToolWrapper = MCPToolWrapper


@dataclass
class MCPServerEntry:
    """Configuration for a single MCP server process.

    Attributes:
        category: Server category name (e.g. 'os_tools').
        command: Executable command (e.g. 'npx').
        args: Command arguments.
        env: Environment variables for the server process.
    """

    category: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    _session: Any = field(default=None, repr=False)
    _stdio_context: Any = field(default=None, repr=False)
    _session_context: Any = field(default=None, repr=False)


@dataclass
class ScoredTool:
    """A tool with a relevance score from semantic matching.

    Attributes:
        tool: The matched tool wrapper.
        score: Cosine similarity score (0.0–1.0).
    """

    tool: MCPToolWrapper
    score: float


class ToolRegistry:
    """Singleton registry of all available MCP tools.

    Loads tool server definitions from servers.yaml, spawns
    MCP stdio server processes on demand, and provides
    lookup, description, and semantic scoring.

    Args:
        config_path: Path to MCP servers config YAML.
    """

    _instance: ToolRegistry | None = None

    def __init__(
        self,
        config_path: str | Path | None = None,
        api_key: str | None = None,
    ) -> None:
        self._config_path = Path(config_path) if config_path else _DEFAULT_CONFIG
        self._api_key = api_key  # kept for backward compat
        self._tools: dict[str, MCPToolWrapper] = {}
        self._servers: dict[str, MCPServerEntry] = {}
        self._tool_to_server: dict[str, str] = {}
        self._native_tools: dict[str, Any] = {}  # name → async callable
        self._embeddings: dict[str, Any] | None = None
        self._embed_model: Any = None
        self._defaults: dict[str, Any] = {}

    @classmethod
    def get_instance(
        cls,
        api_key: str | None = None,
    ) -> ToolRegistry:
        """Get or create the singleton instance.

        Args:
            api_key: Deprecated, kept for backward compat.

        Returns:
            The singleton ToolRegistry.
        """
        if cls._instance is None:
            cls._instance = cls(api_key=api_key)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    def load_from_config(
        self,
        config_path: str | Path | None = None,
    ) -> None:
        """Load MCP server definitions from YAML config.

        Args:
            config_path: Override config path.
        """
        path = Path(config_path) if config_path else self._config_path

        try:
            import yaml

            if not path.exists():
                logger.warning(
                    "mcp_config_not_found",
                    path=str(path),
                )
                return

            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            self._defaults = config.get("defaults", {})
            servers = config.get("servers", {})

            for category, server_config in servers.items():
                # Resolve env var references like ${GITHUB_PAT}
                env = {}
                for key, val in server_config.get("env", {}).items():
                    if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                        env_name = val[2:-1]
                        env[key] = os.environ.get(env_name, "")
                    else:
                        env[key] = str(val)

                cmd_raw = server_config["command"]
                cmd_resolved = cmd_raw
                if cmd_raw == "uvx":
                    cmd_resolved = UVX_BIN
                elif cmd_raw == "npx":
                    cmd_resolved = NPX_BIN

                entry = MCPServerEntry(
                    category=category,
                    command=cmd_resolved,
                    args=server_config.get("args", []),
                    env=env,
                )
                self._servers[category] = entry

            logger.info(
                "mcp_config_loaded",
                server_count=len(self._servers),
                categories=list(self._servers.keys()),
            )

        except Exception as exc:
            logger.error(
                "mcp_config_load_failed",
                error=str(exc),
            )

    def load(
        self,
        enabled_apps: list[str] | None = None,
    ) -> None:
        """Legacy load method — delegates to load_from_config.

        Args:
            enabled_apps: Ignored (backward compat).
        """
        self.load_from_config()

    async def _get_session(self, category: str) -> Any:
        """Get or create an MCP client session for a server category.

        Args:
            category: Server category name.

        Returns:
            Active ClientSession or None if spawn failed.
        """
        entry = self._servers.get(category)
        if entry is None:
            logger.warning(
                "mcp_server_not_configured",
                category=category,
            )
            return None

        if entry._session is not None:
            return entry._session

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command=entry.command,
                args=entry.args,
                env={**os.environ, **entry.env} if entry.env else None,
            )

            # Enter the stdio_client context
            entry._stdio_context = stdio_client(server_params)
            read, write = await entry._stdio_context.__aenter__()

            # Enter the ClientSession context
            entry._session_context = ClientSession(read, write)
            entry._session = await entry._session_context.__aenter__()

            # Initialize the MCP session
            await entry._session.initialize()

            logger.info(
                "mcp_server_connected",
                category=category,
                command=entry.command,
            )

            return entry._session

        except Exception as exc:
            logger.warning(
                "mcp_server_spawn_failed",
                category=category,
                command=entry.command,
                error=str(exc),
            )
            return None

    async def discover_tools(self, category: str) -> list[MCPToolWrapper]:
        """Discover tools from an MCP server.

        Args:
            category: Server category to query.

        Returns:
            List of MCPToolWrapper discovered from the server.
        """
        session = await self._get_session(category)
        if session is None:
            return []

        try:
            response = await session.list_tools()
            tools = []
            for tool in response.tools:
                wrapper = MCPToolWrapper(
                    name=tool.name,
                    description=(tool.description or "")[:200],
                    params_schema=(tool.inputSchema if isinstance(tool.inputSchema, dict) else {}),
                    is_destructive=any(p in tool.name.lower() for p in _DESTRUCTIVE_PATTERNS),
                    category=self._infer_category(tool.name),
                    cost_estimate=0.001,
                    server_category=category,
                )
                self._tools[tool.name] = wrapper
                self._tool_to_server[tool.name] = category
                tools.append(wrapper)

            logger.info(
                "mcp_tools_discovered",
                category=category,
                tool_count=len(tools),
            )
            return tools

        except Exception as exc:
            logger.warning(
                "mcp_tool_discovery_failed",
                category=category,
                error=str(exc),
            )
            return []

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a tool on its MCP server.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            Tool result content.
        """
        category = self._tool_to_server.get(tool_name)
        if category is None:
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not mapped to any server",
                tool_name=tool_name,
                available_tools=list(self._tools.keys()),
            )

        session = await self._get_session(category)
        if session is None:
            raise ToolNotFoundError(
                f"MCP server '{category}' not available",
                tool_name=tool_name,
            )

        result = await session.call_tool(tool_name, arguments)

        # Extract content from MCP result
        if result.content:
            texts = []
            for block in result.content:
                if hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts) if texts else str(result.content)

        return str(result)

    def register(self, tool: MCPToolWrapper) -> None:
        """Manually register a tool wrapper.

        Args:
            tool: Tool wrapper to register.
        """
        self._tools[tool.name] = tool
        # Invalidate embeddings cache
        self._embeddings = None

    def register_native(
        self,
        name: str,
        fn: Any,
        description: str = "",
        params_schema: dict[str, Any] | None = None,
        is_destructive: bool = False,
        category: ToolCategory = ToolCategory.VISION,
    ) -> None:
        """Register a native Python callable as a tool.

        Used for tools that bypass MCP stdio (e.g. vision tools that
        call pyautogui directly).

        Args:
            name: Tool action name.
            fn: Async callable implementing the tool.
            description: Human-readable description.
            params_schema: JSON Schema for parameters.
            is_destructive: Whether this tool mutates state.
            category: Tool category for permission routing.
        """
        self._native_tools[name] = fn
        wrapper = MCPToolWrapper(
            name=name,
            description=description[:200],
            params_schema=params_schema or {},
            is_destructive=is_destructive,
            category=category,
            cost_estimate=0.001,
            server_category="native",
        )
        self._tools[name] = wrapper
        self._embeddings = None

        logger.info(
            "native_tool_registered",
            name=name,
            category=category.value,
            destructive=is_destructive,
        )

    async def call_native(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> Any:
        """Call a native Python tool by name.

        Args:
            tool_name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            Tool result.

        Raises:
            ToolNotFoundError: If the tool is not registered as native.
        """
        fn = self._native_tools.get(tool_name)
        if fn is None:
            raise ToolNotFoundError(
                f"Native tool '{tool_name}' not found",
                tool_name=tool_name,
                available_tools=list(self._native_tools.keys()),
            )
        return await fn(**arguments)

    def is_native_tool(self, tool_name: str) -> bool:
        """Check if a tool is a native Python callable.

        Args:
            tool_name: Tool name to check.

        Returns:
            True if the tool is a native callable.
        """
        return tool_name in self._native_tools

    def register_vision_tools(self) -> None:
        """Register all vision tools from the vision_tools module.

        Loads VISION_TOOL_DEFINITIONS and registers each as a
        native callable tool.
        """
        try:
            from orion.tools.categories.vision_tools import (
                VISION_TOOL_DEFINITIONS,
            )

            for tool_def in VISION_TOOL_DEFINITIONS:
                self.register_native(
                    name=tool_def["name"],
                    fn=tool_def["fn"],
                    description=tool_def.get("description", ""),
                    params_schema=tool_def.get("params_schema"),
                    is_destructive=tool_def.get("is_destructive", False),
                    category=ToolCategory.VISION,
                )

            logger.info(
                "vision_tools_registered",
                count=len(VISION_TOOL_DEFINITIONS),
            )

        except ImportError as exc:
            logger.warning(
                "vision_tools_import_failed",
                error=str(exc),
            )

    def register_os_tools(self) -> None:
        """Register all native OS tools.

        Loads NATIVE_OS_TOOL_DEFINITIONS and registers each as a
        native callable tool.
        """
        try:
            from orion.tools.categories.os_tools_native import (
                NATIVE_OS_TOOL_DEFINITIONS,
            )

            for tool_def in NATIVE_OS_TOOL_DEFINITIONS:
                self.register_native(
                    name=tool_def["name"],
                    fn=tool_def["fn"],
                    description=tool_def.get("description", ""),
                    params_schema=tool_def.get("params_schema"),
                    is_destructive=tool_def.get("is_destructive", False),
                    category=ToolCategory.OS,
                )

            logger.info(
                "os_tools_registered",
                count=len(NATIVE_OS_TOOL_DEFINITIONS),
            )

        except ImportError as exc:
            logger.warning(
                "os_tools_import_failed",
                error=str(exc),
            )

    def get(self, name: str) -> MCPToolWrapper:
        """Look up a tool by name.

        Args:
            name: Exact tool action name.

        Returns:
            The matching MCPToolWrapper.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                f"Tool '{name}' not found in registry",
                tool_name=name,
                available_tools=list(self._tools.keys()),
            )
        return tool

    def describe_all(self) -> str:
        """Generate compact tool descriptions for LLM prompts.

        Returns:
            Newline-separated tool descriptions.
        """
        lines = []
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
            destructive = "T" if tool.is_destructive else "F"
            lines.append(
                f"{tool.name}: {tool.description} "
                f"({tool.category.value}, "
                f"destructive={destructive})"
            )
        return "\n".join(lines)

    def score(self, subtask_desc: str, top_k: int = 5) -> list[ScoredTool]:
        """Score tools by semantic similarity to a subtask.

        Uses sentence-transformers all-MiniLM-L6-v2 for embedding.
        Falls back to keyword matching if the model is unavailable.

        Args:
            subtask_desc: Natural language subtask description.
            top_k: Number of top results to return.

        Returns:
            List of ScoredTool sorted by descending score.
        """
        if not self._tools:
            return []

        try:
            return self._semantic_score(subtask_desc, top_k)
        except Exception:
            return self._keyword_score(subtask_desc, top_k)

    def list_tools(self) -> list[MCPToolWrapper]:
        """Return all registered tools.

        Returns:
            List of all tool wrappers.
        """
        return list(self._tools.values())

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def _infer_category(self, name: str) -> ToolCategory:
        """Infer tool category from its action name.

        Args:
            name: Tool action name.

        Returns:
            Inferred ToolCategory.
        """
        upper = name.upper()
        if upper.startswith("GITHUB"):
            return ToolCategory.GITHUB
        if any(
            upper.startswith(p)
            for p in (
                "SHELL",
                "FILE",
                "OS",
                "CMD",
                "READ",
                "WRITE",
                "LIST",
                "SEARCH",
                "CREATE_DIR",
                "MOVE",
            )
        ):
            return ToolCategory.OS
        if upper.startswith("BROWSER"):
            return ToolCategory.BROWSER
        if any(
            upper.startswith(p)
            for p in (
                "TAKE_SCREENSHOT",
                "ANALYZE_SCREEN",
                "CLICK_ELEMENT",
                "TYPE_TEXT",
                "PRESS_KEY",
                "VISION",
            )
        ):
            return ToolCategory.VISION
        if any(upper.startswith(p) for p in ("SLACK", "LINEAR", "NOTION", "GMAIL", "CALENDAR")):
            return ToolCategory.SAAS
        return ToolCategory.SYSTEM

    def _wrap_tool(self, raw_tool: Any) -> MCPToolWrapper:
        """Wrap a raw tool object into our wrapper format.

        Args:
            raw_tool: Raw tool object with name/description/parameters.

        Returns:
            MCPToolWrapper instance.
        """
        name = getattr(raw_tool, "name", str(raw_tool))
        description = getattr(raw_tool, "description", "No description")
        params_schema = getattr(
            raw_tool,
            "parameters",
            getattr(raw_tool, "inputSchema", {}),
        )

        # Detect destructiveness from name
        name_lower = name.lower()
        is_destructive = any(p in name_lower for p in _DESTRUCTIVE_PATTERNS)

        # Infer category from name
        category = self._infer_category(name)

        return MCPToolWrapper(
            name=name,
            description=description[:200],
            params_schema=(params_schema if isinstance(params_schema, dict) else {}),
            is_destructive=is_destructive,
            category=category,
            cost_estimate=0.001,
        )

    def _semantic_score(self, desc: str, top_k: int) -> list[ScoredTool]:
        """Score via sentence-transformer embeddings.

        Args:
            desc: Subtask description.
            top_k: Number of results.

        Returns:
            Scored tool list.
        """
        if self._embed_model is None:
            from sentence_transformers import (
                SentenceTransformer,
            )

            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        if self._embeddings is None:
            texts = [f"{t.name} {t.description}" for t in self._tools.values()]
            self._embeddings = self._embed_model.encode(texts)

        import numpy as np

        query_emb = self._embed_model.encode([desc])[0]
        scores = []
        tools_list = list(self._tools.values())

        for i, tool in enumerate(tools_list):
            emb = self._embeddings[i]
            cos_sim = float(
                np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)
            )
            scores.append(ScoredTool(tool=tool, score=cos_sim))

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[:top_k]

    def _keyword_score(self, desc: str, top_k: int) -> list[ScoredTool]:
        """Fallback keyword-based scoring.

        Args:
            desc: Subtask description.
            top_k: Number of results.

        Returns:
            Scored tool list.
        """
        desc_words = set(desc.lower().split())
        scored = []

        for tool in self._tools.values():
            tool_words = set(f"{tool.name} {tool.description}".lower().split())
            overlap = len(desc_words & tool_words)
            total = len(desc_words | tool_words) or 1
            score = overlap / total
            scored.append(ScoredTool(tool=tool, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
