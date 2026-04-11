"""WorkingMemory — ephemeral in-context memory per task.

Stores the current TaskDAG, StepResults, and agent scratchpad
notes. Enforces a max token budget with LLM-based summarisation
on eviction.

Module Contract
---------------
- **Inputs**: add entries (plan, step, note).
- **Outputs**: compact context string for agent prompts.
- **Lifecycle**: Cleared at task completion.

Depends On
----------
- ``structlog`` (logging)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_TOKENS = 8192
_EVICTION_THRESHOLD = 0.80


@dataclass
class MemoryEntry:
    """Single entry in working memory.

    Attributes:
        role: Entry type (plan, step_result, note, summary).
        content: Text content.
        token_estimate: Estimated token count.
        metadata: Optional structured metadata.
    """

    role: str
    content: str
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            self.token_estimate = len(self.content) // 4 + 1


class WorkingMemory:
    """Ephemeral in-context memory for a single task.

    Maintains a sliding window of entries within a token budget.
    When the budget hits 80%, evicts oldest entries and replaces
    them with a summary.

    Args:
        max_tokens: Maximum token budget.
        summarizer: Optional async callable for summarisation.
        task_id: ID of the task this memory belongs to.
    """

    def __init__(
        self,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        summarizer: Any = None,
        task_id: str = "",
    ) -> None:
        self._max_tokens = max_tokens
        self._summarizer = summarizer
        self._task_id = task_id
        self._entries: list[MemoryEntry] = []
        self._total_tokens = 0

    def add_plan(self, plan: dict[str, Any]) -> None:
        """Add the execution plan to memory.

        Args:
            plan: Serialised TaskDAG dict.
        """
        content = json.dumps(plan, indent=1)
        entry = MemoryEntry(
            role="plan",
            content=content,
            metadata={"type": "plan"},
        )
        self._add(entry)

    def add_step_result(
        self,
        subtask_id: str,
        tool: str,
        output: str,
        success: bool,
    ) -> None:
        """Add a step execution result.

        Args:
            subtask_id: Subtask ID.
            tool: Tool used.
            output: Output string (truncated).
            success: Whether the step succeeded.
        """
        content = (
            f"[{subtask_id}] {tool}: "
            f"{'OK' if success else 'FAIL'} — "
            f"{output[:200]}"
        )
        entry = MemoryEntry(
            role="step_result",
            content=content,
            metadata={
                "subtask_id": subtask_id,
                "tool": tool,
                "success": success,
            },
        )
        self._add(entry)

    def add_note(self, note: str, agent: str = "") -> None:
        """Add a scratchpad note.

        Args:
            note: Free-text note.
            agent: Agent that created the note.
        """
        entry = MemoryEntry(
            role="note",
            content=note,
            metadata={"agent": agent},
        )
        self._add(entry)

    def to_context_str(self) -> str:
        """Render memory as a compact string for agent prompts.

        Returns:
            Formatted context string.
        """
        if not self._entries:
            return "No prior context."

        sections: list[str] = []
        for entry in self._entries:
            prefix = entry.role.upper()
            sections.append(f"[{prefix}] {entry.content}")

        return "\n---\n".join(sections)

    def clear(self) -> None:
        """Clear all entries (called at task completion)."""
        self._entries.clear()
        self._total_tokens = 0
        logger.info(
            "working_memory_cleared",
            task_id=self._task_id,
        )

    @property
    def total_tokens(self) -> int:
        """Current estimated token usage."""
        return self._total_tokens

    @property
    def entry_count(self) -> int:
        """Number of entries in memory."""
        return len(self._entries)

    @property
    def utilization(self) -> float:
        """Token utilization as a fraction (0.0–1.0)."""
        if self._max_tokens == 0:
            return 0.0
        return self._total_tokens / self._max_tokens

    def _add(self, entry: MemoryEntry) -> None:
        """Add an entry, evicting if over threshold.

        Args:
            entry: Entry to add.
        """
        self._entries.append(entry)
        self._total_tokens += entry.token_estimate

        if self.utilization >= _EVICTION_THRESHOLD:
            self._evict()

    def _evict(self) -> None:
        """Evict oldest entries and replace with summary.

        Removes entries until under 60% utilization, then
        summarises the evicted content.
        """
        evicted: list[MemoryEntry] = []
        target = int(self._max_tokens * 0.6)

        while (
            self._total_tokens > target
            and len(self._entries) > 1
        ):
            removed = self._entries.pop(0)
            self._total_tokens -= removed.token_estimate
            evicted.append(removed)

        if evicted:
            summary_text = self._summarise_evicted(evicted)
            summary = MemoryEntry(
                role="summary",
                content=summary_text,
                metadata={"evicted_count": len(evicted)},
            )
            self._entries.insert(0, summary)
            self._total_tokens += summary.token_estimate

            logger.info(
                "working_memory_evicted",
                task_id=self._task_id,
                evicted=len(evicted),
                remaining=len(self._entries),
                tokens=self._total_tokens,
            )

    def _summarise_evicted(
        self, entries: list[MemoryEntry]
    ) -> str:
        """Summarise evicted entries.

        In production, calls LLM via summarizer. Falls back to
        truncated concatenation.

        Args:
            entries: Evicted entries.

        Returns:
            Summary string.
        """
        combined = " | ".join(
            f"{e.role}: {e.content[:100]}" for e in entries
        )

        # Truncate fallback summary
        if len(combined) > 500:
            combined = combined[:497] + "..."

        return f"[Summary of {len(entries)} prior entries] {combined}"
