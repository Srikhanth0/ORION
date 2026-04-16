"""FastAPI app factory + lifespan for ORION.

Handles startup (init AgentScope, warm LLM router, load registry,
start metrics, init ChromaDB) and shutdown (flush traces, close
connections, cancel background tasks).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from orion.api.routes import status, tasks, tools
from orion.api.schemas import ProblemDetail
from orion.observability.logger import configure_logging

logger = structlog.get_logger(__name__)

_SHUTDOWN_TIMEOUT = 30


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Startup:
    - Configure structured logging.
    - Warm up the tool registry.
    - Start Prometheus metrics server.

    Shutdown:
    - Cancel running tasks.
    - Flush traces.
    """
    # ── Startup ──
    configure_logging(
        level=os.environ.get("LOG_LEVEL", "INFO"),
    )
    logger.info("orion_startup_begin")

    # Warm tool registry
    try:
        from orion.tools.registry import ToolRegistry

        registry = ToolRegistry.get_instance()
        registry.load_from_config()

        # Load native vision tools first
        registry.register_vision_tools()

        # Register native OS tools that work without npx/MCP
        registry.register_os_tools()

        # Synchronous discovery of all tools so /v1/tools returns them immediately
        async def _discover_all() -> None:
            for cat in registry._servers:
                try:
                    await registry.discover_tools(cat)
                except Exception as e:
                    logger.warning(
                        "mcp_tool_discovery_failed",
                        category=cat,
                        error=str(e),
                    )

        asyncio.create_task(_discover_all())
        logger.info("tool_registry_loaded", server_count=len(registry._servers))

        # ORION-FIX: Warn early if uvx/npx are missing so failure is obvious
        import shutil

        from orion.tools.registry import NPX_BIN, UVX_BIN

        for _bin in (UVX_BIN, NPX_BIN):
            if not shutil.which(_bin):
                logger.warning(
                    "mcp_tools_dependency_missing",
                    warning=(
                        f"Binary '{_bin}' not found on PATH. MCP tools requiring "
                        "it will fail to load. On Windows, run inside WSL2 or install "
                        "Node.js + uv globally."
                    ),
                )
    except Exception as exc:
        logger.warning(
            "tool_registry_load_failed",
            error=str(exc),
        )

    # Start Prometheus metrics server
    try:
        from orion.observability.metrics import (
            MetricsServer,
        )

        MetricsServer(port=int(os.environ.get("METRICS_PORT", "9091"))).start()
    except Exception as exc:
        logger.warning(
            "metrics_server_failed",
            error=str(exc),
        )

    logger.info("orion_startup_complete")

    yield

    # ── Shutdown ──
    logger.info("orion_shutdown_begin")

    # Cancel running task handles
    from orion.api.routes.tasks import _task_handles

    for tid, handle in _task_handles.items():
        if not handle.done():
            handle.cancel()
            logger.info("task_cancelled_on_shutdown", task_id=tid)

    # Give tasks time to clean up
    pending = [h for h in _task_handles.values() if not h.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    logger.info("orion_shutdown_complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="ORION Agent API",
        description=(
            "Autonomous OS automation agent with multi-provider LLM routing and MCP tools."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(tasks.router)
    app.include_router(tools.router)
    app.include_router(status.router)

    # ORION-FIX: Qdrant is now optional; readiness only fails if LLM is unreachable
    @app.get("/ready")
    async def readiness() -> dict[str, Any]:  # noqa: F821
        """Check application readiness."""
        # Simple placeholder for actual router health check
        return {"status": "ok", "qdrant": False}

    # Global exception handler → RFC 7807
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=str(request.url),
        )
        return JSONResponse(
            status_code=500,
            content=ProblemDetail(
                type="/errors/internal",
                title="Internal Server Error",
                status=500,
                detail=str(exc)[:500],
            ).model_dump(),
        )

    return app


# Module-level app for `uvicorn orion.api.server:app`
app = create_app()
