"""ToolRegistry — singleton index of all available Composio tools.

Loads tools from the Composio SDK at startup, wraps each as a
``ComposioToolWrapper``, and provides lookup, description, and
semantic scoring for the Planner agent.

Module Contract
---------------
- **Inputs**: Composio API key + list of enabled apps.
- **Outputs**: Tool lookup, compact descriptions, scored suggestions.

Depends On
----------
- ``composio`` SDK (ComposioToolSet)
- ``orion.core.exceptions`` (ToolNotFoundError)
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import structlog

from orion.core.exceptions import ToolNotFoundError

logger = structlog.get_logger(__name__)


class ToolCategory(enum.StrEnum):
    """Tool category classification."""

    GITHUB = "github"
    OS = "os"
    BROWSER = "browser"
    SAAS = "saas"
    SYSTEM = "system"


_DESTRUCTIVE_PATTERNS = frozenset({
    "delete", "remove", "kill", "drop", "destroy",
    "purge", "truncate", "wipe", "erase", "uninstall",
})


@dataclass(frozen=True)
class ComposioToolWrapper:
    """Immutable wrapper around a single Composio tool action.

    Attributes:
        name: Composio action name (e.g. ``GITHUB_CREATE_ISSUE``).
        description: Human-readable description from the tool.
        params_schema: JSON Schema dict for parameter validation.
        is_destructive: Whether this tool mutates external state.
        category: Tool category for permission routing.
        cost_estimate: Estimated token cost per average call.
    """

    name: str
    description: str
    params_schema: dict[str, Any] = field(default_factory=dict)
    is_destructive: bool = False
    category: ToolCategory = ToolCategory.SYSTEM
    cost_estimate: float = 0.0


@dataclass
class ScoredTool:
    """A tool with a relevance score from semantic matching.

    Attributes:
        tool: The matched tool wrapper.
        score: Cosine similarity score (0.0–1.0).
    """

    tool: ComposioToolWrapper
    score: float


class ToolRegistry:
    """Singleton registry of all available Composio tools.

    Loads tools from the Composio API, wraps them as
    ``ComposioToolWrapper`` objects, and provides lookup,
    description generation, and semantic scoring.

    Args:
        api_key: Composio API key (optional, from env).
    """

    _instance: ToolRegistry | None = None

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._tools: dict[str, ComposioToolWrapper] = {}
        self._embeddings: dict[str, Any] | None = None
        self._embed_model: Any = None

    @classmethod
    def get_instance(
        cls, api_key: str | None = None
    ) -> ToolRegistry:
        """Get or create the singleton instance.

        Args:
            api_key: Composio API key.

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

    def load(
        self, enabled_apps: list[str] | None = None
    ) -> None:
        """Fetch tools from Composio and build the internal index.

        Args:
            enabled_apps: List of Composio app names to load.
        """
        try:
            from composio import ComposioToolSet

            toolset = ComposioToolSet(api_key=self._api_key)

            tools = toolset.get_tools(apps=enabled_apps) if enabled_apps else toolset.get_tools()

            for tool in tools:
                wrapper = self._wrap_tool(tool)
                self._tools[wrapper.name] = wrapper

            logger.info(
                "tool_registry_loaded",
                tool_count=len(self._tools),
                apps=enabled_apps,
            )

        except ImportError:
            logger.warning("composio_sdk_not_available")
        except Exception as exc:
            logger.error(
                "tool_registry_load_failed",
                error=str(exc),
            )

    def register(self, tool: ComposioToolWrapper) -> None:
        """Manually register a tool wrapper.

        Args:
            tool: Tool wrapper to register.
        """
        self._tools[tool.name] = tool
        # Invalidate embeddings cache
        self._embeddings = None

    def get(self, name: str) -> ComposioToolWrapper:
        """Look up a tool by name.

        Args:
            name: Exact tool action name.

        Returns:
            The matching ComposioToolWrapper.

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
        for tool in sorted(
            self._tools.values(), key=lambda t: t.name
        ):
            destructive = "T" if tool.is_destructive else "F"
            lines.append(
                f"{tool.name}: {tool.description} "
                f"({tool.category.value}, "
                f"destructive={destructive})"
            )
        return "\n".join(lines)

    def score(
        self, subtask_desc: str, top_k: int = 5
    ) -> list[ScoredTool]:
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

    def list_tools(self) -> list[ComposioToolWrapper]:
        """Return all registered tools.

        Returns:
            List of all tool wrappers.
        """
        return list(self._tools.values())

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def _wrap_tool(self, raw_tool: Any) -> ComposioToolWrapper:
        """Wrap a raw Composio tool into our wrapper format.

        Args:
            raw_tool: Raw tool object from Composio SDK.

        Returns:
            ComposioToolWrapper instance.
        """
        name = getattr(raw_tool, "name", str(raw_tool))
        description = getattr(
            raw_tool, "description", "No description"
        )
        params_schema = getattr(
            raw_tool, "parameters", {}
        )

        # Detect destructiveness from name
        name_lower = name.lower()
        is_destructive = any(
            p in name_lower for p in _DESTRUCTIVE_PATTERNS
        )

        # Infer category from name
        category = self._infer_category(name)

        return ComposioToolWrapper(
            name=name,
            description=description[:200],
            params_schema=(
                params_schema if isinstance(params_schema, dict)
                else {}
            ),
            is_destructive=is_destructive,
            category=category,
            cost_estimate=0.001,
        )

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
            for p in ("SHELL", "FILE", "OS", "CMD")
        ):
            return ToolCategory.OS
        if upper.startswith("BROWSER"):
            return ToolCategory.BROWSER
        if any(
            upper.startswith(p)
            for p in ("SLACK", "LINEAR", "NOTION", "GMAIL",
                       "CALENDAR")
        ):
            return ToolCategory.SAAS
        return ToolCategory.SYSTEM

    def _semantic_score(
        self, desc: str, top_k: int
    ) -> list[ScoredTool]:
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
            self._embed_model = SentenceTransformer(
                "all-MiniLM-L6-v2"
            )

        if self._embeddings is None:
            texts = [
                f"{t.name} {t.description}"
                for t in self._tools.values()
            ]
            self._embeddings = (
                self._embed_model.encode(texts)
            )

        import numpy as np

        query_emb = self._embed_model.encode([desc])[0]
        scores = []
        tools_list = list(self._tools.values())

        for i, tool in enumerate(tools_list):
            emb = self._embeddings[i]
            cos_sim = float(
                np.dot(query_emb, emb)
                / (
                    np.linalg.norm(query_emb)
                    * np.linalg.norm(emb)
                    + 1e-8
                )
            )
            scores.append(ScoredTool(tool=tool, score=cos_sim))

        scores.sort(key=lambda s: s.score, reverse=True)
        return scores[:top_k]

    def _keyword_score(
        self, desc: str, top_k: int
    ) -> list[ScoredTool]:
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
            tool_words = set(
                f"{tool.name} {tool.description}".lower().split()
            )
            overlap = len(desc_words & tool_words)
            total = len(desc_words | tool_words) or 1
            score = overlap / total
            scored.append(ScoredTool(tool=tool, score=score))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
