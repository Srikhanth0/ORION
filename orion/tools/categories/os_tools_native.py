"""Native OS tools — direct Python implementations without MCP.

These tools work without npx/node and provide core file operations.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any


async def list_directory(path: str = ".") -> dict[str, Any]:
    """List files and directories in a path.
    
    Args:
        path: Directory to list (default: current dir)
        
    Returns:
        Dict with files list and metadata
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}
    if not p.is_dir():
        return {"error": f"Path is not a directory: {path}"}
    
    items = []
    for item in p.iterdir():
        items.append({
            "name": item.name,
            "type": "dir" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else 0,
        })
    
    return {"path": str(p), "items": items, "count": len(items)}


async def read_text_file(path: str) -> dict[str, Any]:
    """Read text file contents.
    
    Args:
        path: File path to read
        
    Returns:
        Dict with file contents or error
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"File does not exist: {path}"}
    if not p.is_file():
        return {"error": f"Path is not a file: {path}"}
    
    try:
        content = p.read_text(encoding="utf-8")
        return {"path": str(p), "content": content, "size": len(content)}
    except Exception as e:
        return {"error": f"Read error: {e}"}


async def write_file(path: str, content: str) -> dict[str, Any]:
    """Write/create a file with content.
    
    Args:
        path: File path to write
        content: Text content to write
        
    Returns:
        Dict with success status or error
    """
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "written": len(content)}
    except Exception as e:
        return {"error": f"Write error: {e}"}


async def create_directory(path: str) -> dict[str, Any]:
    """Create a directory.
    
    Args:
        path: Directory path to create
        
    Returns:
        Dict with success status or error
    """
    p = Path(path).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        return {"path": str(p), "created": True}
    except Exception as e:
        return {"error": f"Create error: {e}"}


async def search_files(path: str = ".", pattern: str = "*") -> dict[str, Any]:
    """Search for files matching a pattern.
    
    Args:
        path: Directory to search in
        pattern: Glob pattern (e.g., "*.py")
        
    Returns:
        Dict with matching files
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}
    
    matches = []
    for m in p.glob(pattern):
        matches.append(str(m))
    
    return {"path": str(p), "pattern": pattern, "matches": matches, "count": len(matches)}


async def get_file_info(path: str) -> dict[str, Any]:
    """Get file/directory metadata.
    
    Args:
        path: File or directory path
        
    Returns:
        Dict with metadata
    """
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}
    
    stat = p.stat()
    return {
        "path": str(p),
        "type": "dir" if p.is_dir() else "file",
        "size": stat.st_size,
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
    }


async def move_file(src: str, dst: str) -> dict[str, Any]:
    """Move/rename a file or directory.
    
    Args:
        src: Source path
        dst: Destination path
        
    Returns:
        Dict with success status
    """
    src_p = Path(src).expanduser()
    dst_p = Path(dst).expanduser()
    
    try:
        shutil.move(str(src_p), str(dst_p))
        return {"from": str(src_p), "to": str(dst_p), "moved": True}
    except Exception as e:
        return {"error": f"Move error: {e}"}


# Tool definitions for registry
NATIVE_OS_TOOL_DEFINITIONS = [
    {
        "name": "list_directory",
        "description": "List files and directories in a path",
        "fn": list_directory,
        "is_destructive": False,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"}
            }
        },
    },
    {
        "name": "read_text_file",
        "description": "Read the complete contents of a text file",
        "fn": read_text_file,
        "is_destructive": False,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with content",
        "fn": write_file,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "File content"}
            },
            "required": ["path", "content"]
        },
    },
    {
        "name": "create_directory",
        "description": "Create a new directory",
        "fn": create_directory,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path"}
            },
            "required": ["path"]
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern",
        "fn": search_files,
        "is_destructive": False,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory to search"},
                "pattern": {"type": "string", "description": "Glob pattern"}
            }
        },
    },
    {
        "name": "get_file_info",
        "description": "Get file or directory metadata",
        "fn": get_file_info,
        "is_destructive": False,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File/directory path"}
            },
            "required": ["path"]
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename a file or directory",
        "fn": move_file,
        "is_destructive": True,
        "params_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string", "description": "Source path"},
                "dst": {"type": "string", "description": "Destination path"}
            },
            "required": ["src", "dst"]
        },
    },
]