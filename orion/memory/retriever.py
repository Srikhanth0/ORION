"""Retriever — facade for memory retrieval across tiers.

Combines WorkingMemory context with LongTermMemory retrieval
into a single interface for agent prompts.

Module Contract
---------------
- **Inputs**: query string.
- **Outputs**: combined context from both tiers.

Depends On
----------
- ``orion.memory.working`` (WorkingMemory)
- ``orion.memory.longterm`` (LongTermMemory, PastTask)
"""

from __future__ import annotations

import structlog

from orion.memory.longterm import LongTermMemory, PastTask
from orion.memory.working import WorkingMemory

logger = structlog.get_logger(__name__)


class MemoryRetriever:
    """Unified retrieval across working and long-term memory.

    Combines in-context working memory with semantic search
    over the long-term ChromaDB store.

    Args:
        working: WorkingMemory for the current task.
        longterm: LongTermMemory for cross-task retrieval.
    """

    def __init__(
        self,
        working: WorkingMemory | None = None,
        longterm: LongTermMemory | None = None,
    ) -> None:
        self._working = working
        self._longterm = longterm

    def get_context(
        self,
        query: str = "",
        include_longterm: bool = True,
        top_k: int = 3,
    ) -> str:
        """Get combined context from both memory tiers.

        Args:
            query: Optional query for long-term retrieval.
            include_longterm: Whether to include past tasks.
            top_k: Max past tasks to retrieve.

        Returns:
            Combined context string.
        """
        sections: list[str] = []

        # Tier 1: Working memory
        if self._working is not None:
            ctx = self._working.to_context_str()
            if ctx and ctx != "No prior context.":
                sections.append("=== WORKING MEMORY ===\n" + ctx)

        # Tier 2: Long-term memory
        if include_longterm and self._longterm is not None and query:
            past = self._longterm.retrieve(query, top_k=top_k)
            if past:
                lt_text = self._format_past_tasks(past)
                sections.append("=== PAST TASKS ===\n" + lt_text)

        if not sections:
            return "No prior context available."

        return "\n\n".join(sections)

    def get_past_tasks(self, query: str, top_k: int = 3) -> list[PastTask]:
        """Retrieve similar past tasks.

        Args:
            query: Search query.
            top_k: Max results.

        Returns:
            List of PastTask.
        """
        if self._longterm is None:
            return []
        return self._longterm.retrieve(query, top_k=top_k)

    def _format_past_tasks(self, tasks: list[PastTask]) -> str:
        """Format past tasks for prompt injection.

        Args:
            tasks: Retrieved past tasks.

        Returns:
            Formatted string.
        """
        lines: list[str] = []
        for i, t in enumerate(tasks, 1):
            status = "✓" if t.success else "✗"
            tools = ", ".join(t.tools_used[:5]) or "none"
            lines.append(
                f"{i}. [{status}] {t.task_description[:100]}"
                f" (score={t.score:.2f}, "
                f"tools=[{tools}], "
                f"{t.duration_seconds:.1f}s)"
            )
            if t.step_results_summary:
                lines.append(f"   Summary: {t.step_results_summary[:150]}")
        return "\n".join(lines)
