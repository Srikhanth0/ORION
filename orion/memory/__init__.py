"""ORION memory — two-tier memory architecture.

Exports
-------
- ``Embedder`` — singleton semantic embedding pipeline
- ``WorkingMemory`` — ephemeral in-context memory per task
- ``LongTermMemory`` — persistent Qdrant-backed cross-task memory
- ``MemoryRetriever`` — unified retrieval facade
- ``PastTask`` — retrieved task dataclass
"""
from __future__ import annotations

from orion.memory.embedder import Embedder
from orion.memory.longterm import LongTermMemory, PastTask
from orion.memory.retriever import MemoryRetriever
from orion.memory.working import MemoryEntry, WorkingMemory

__all__: list[str] = [
    "Embedder",
    "LongTermMemory",
    "MemoryEntry",
    "MemoryRetriever",
    "PastTask",
    "WorkingMemory",
]
