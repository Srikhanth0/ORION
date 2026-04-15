"""ORION tool category wrappers.

Thin typed wrappers per tool category that map to underlying
MCP action names (or native Python callables for vision tools)
with Python docstrings and type hints.
"""
from __future__ import annotations

__all__: list[str] = [
    "github_tools",
    "os_tools",
    "browser_tools",
    "saas_tools",
    "vision_tools",
]
