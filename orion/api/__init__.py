"""ORION API — external-facing FastAPI server.

Provides REST endpoints for task submission, status polling,
SSE streaming, and tool registry inspection.

Exports
-------
- ``create_app`` — FastAPI app factory
- ``app`` — Module-level app instance

Submodules
----------
- ``server`` — FastAPI app factory + lifespan
- ``schemas`` — Pydantic request/response models
- ``routes/tasks`` — Task CRUD + SSE streaming
- ``routes/tools`` — Tool registry browsing
- ``routes/status`` — Health/readiness probes

Depends On
----------
- ``fastapi``, ``uvicorn``, ``sse_starlette``
- ``orion.agents`` (pipeline execution)
- ``orion.tools`` (registry inspection)
- ``orion.observability`` (logging, metrics)
"""
from __future__ import annotations

from orion.api.server import app, create_app

__all__: list[str] = ["app", "create_app"]
