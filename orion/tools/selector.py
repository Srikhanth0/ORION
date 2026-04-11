"""ToolSelector — semantic tool matching for the Planner agent.

Thin facade over ToolRegistry.score() that provides a Planner-friendly
interface for selecting the best tools for a given subtask.

Module Contract
---------------
- **Inputs**: Subtask description string.
- **Outputs**: Ranked list of tool suggestions with scores.

Depends On
----------
- ``orion.tools.registry`` (ToolRegistry, ScoredTool)
"""
from __future__ import annotations

from dataclasses import dataclass

import structlog

from orion.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)


@dataclass
class ToolSuggestion:
    """A tool suggestion with metadata for the Planner.

    Attributes:
        name: Tool action name.
        description: Tool description.
        score: Relevance score (0.0–1.0).
        is_destructive: Whether the tool is destructive.
        category: Tool category string.
    """

    name: str
    description: str
    score: float
    is_destructive: bool
    category: str


class ToolSelector:
    """Selects the best tools for a given subtask description.

    Wraps ToolRegistry.score() with a Planner-friendly interface,
    filtering and formatting results for prompt injection.

    Args:
        registry: ToolRegistry instance.
        min_score: Minimum score threshold for inclusion.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        min_score: float = 0.1,
    ) -> None:
        self._registry = registry or ToolRegistry.get_instance()
        self._min_score = min_score

    def suggest(
        self,
        subtask_desc: str,
        top_k: int = 5,
    ) -> list[ToolSuggestion]:
        """Suggest tools for a subtask description.

        Args:
            subtask_desc: Natural language subtask description.
            top_k: Maximum number of suggestions.

        Returns:
            Ranked list of ToolSuggestion objects.
        """
        scored = self._registry.score(subtask_desc, top_k=top_k)

        suggestions = []
        for st in scored:
            if st.score >= self._min_score:
                suggestions.append(
                    ToolSuggestion(
                        name=st.tool.name,
                        description=st.tool.description,
                        score=round(st.score, 4),
                        is_destructive=st.tool.is_destructive,
                        category=st.tool.category.value,
                    )
                )

        logger.debug(
            "tool_suggestions",
            subtask=subtask_desc[:80],
            count=len(suggestions),
        )

        return suggestions

    def format_for_prompt(
        self,
        subtask_desc: str,
        top_k: int = 5,
    ) -> str:
        """Format tool suggestions as a string for LLM prompts.

        Args:
            subtask_desc: Natural language subtask description.
            top_k: Maximum number of suggestions.

        Returns:
            Formatted string of tool suggestions.
        """
        suggestions = self.suggest(subtask_desc, top_k=top_k)

        if not suggestions:
            return "No matching tools found."

        lines = ["Suggested tools (ranked by relevance):"]
        for i, s in enumerate(suggestions, 1):
            destr = " [DESTRUCTIVE]" if s.is_destructive else ""
            lines.append(
                f"  {i}. {s.name} (score={s.score}, "
                f"{s.category}{destr}): {s.description}"
            )

        return "\n".join(lines)
