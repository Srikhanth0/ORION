"""Status routes — GET /health, GET /ready, GET /metrics."""

from __future__ import annotations

import structlog
from fastapi import APIRouter

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["status"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200.

    Returns:
        Simple status dict.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    """Readiness probe — checks all dependencies.

    Returns:
        Status dict with dependency health.
    """
    checks: dict[str, bool] = {}

    # Check ChromaDB
    try:
        import chromadb

        client = chromadb.PersistentClient(path=".orion_memory")
        client.heartbeat()
        checks["chromadb"] = True
    except Exception:
        checks["chromadb"] = False

    # Check tool registry
    try:
        from orion.tools.registry import ToolRegistry

        reg = ToolRegistry.get_instance()
        checks["tool_registry"] = reg.tool_count > 0
    except Exception:
        checks["tool_registry"] = False

    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks,
    }
