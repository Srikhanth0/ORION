"""Tool routes — GET /v1/tools, GET /v1/tools/{name}."""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from orion.api.schemas import ProblemDetail, ToolSchema
from orion.tools.registry import ToolRegistry

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1/tools", tags=["tools"])


@router.get("", response_model=list[ToolSchema])
async def list_tools() -> list[ToolSchema]:
    """List all registered MCP tools."""
    try:
        registry = ToolRegistry.get_instance()
    except Exception:
        return []

    return [
        ToolSchema(
            name=t.name,
            description=t.description,
            category=t.category.value,
            is_destructive=t.is_destructive,
            params_schema=t.params_schema,
        )
        for t in registry.list_tools()
    ]


@router.get(
    "/{tool_name}", response_model=ToolSchema
)
async def get_tool(tool_name: str) -> ToolSchema:
    """Get a single tool schema by name."""
    try:
        registry = ToolRegistry.get_instance()
        tool = registry.get(tool_name)
    except Exception as exc:
        raise HTTPException(
            status_code=404,
            detail=ProblemDetail(
                type="/errors/tool-not-found",
                title="Tool Not Found",
                status=404,
                detail=str(exc),
            ).model_dump(),
        ) from exc

    return ToolSchema(
        name=tool.name,
        description=tool.description,
        category=tool.category.value,
        is_destructive=tool.is_destructive,
        params_schema=tool.params_schema,
    )
