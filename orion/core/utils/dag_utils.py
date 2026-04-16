"""DAG and topological sort utilities."""

from __future__ import annotations

from typing import Any


def topological_sort(subtasks: list[dict[str, Any]]) -> list[str]:
    """Topological sort of subtask IDs respecting depends_on.

    Ported from ExecutorAgent for centralized reuse.

    Args:
        subtasks: List of subtask dicts with 'id' and 'depends_on'.

    Returns:
        Ordered list of subtask IDs.
    """
    graph: dict[str, list[str]] = {}
    in_degree: dict[str, int] = {}
    for t in subtasks:
        tid = t["id"]
        graph.setdefault(tid, [])
        in_degree.setdefault(tid, 0)
        for dep in t.get("depends_on", []):
            graph.setdefault(dep, []).append(tid)
            in_degree[tid] = in_degree.get(tid, 0) + 1

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    ordered: list[str] = []
    while queue:
        node = queue.pop(0)
        ordered.append(node)
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return ordered
