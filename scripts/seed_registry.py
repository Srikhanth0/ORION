#!/usr/bin/env python3
"""ORION seed_registry — discover tools from MCP servers.

Reads MCP server definitions from configs/mcp/servers.yaml,
spawns each server process, and lists all discovered tools.

Usage:
    python scripts/seed_registry.py
    python scripts/seed_registry.py --config configs/mcp/servers.yaml
    python scripts/seed_registry.py --export tools.json
    python scripts/seed_registry.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Discover and list tools from MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/seed_registry.py                    # Discover all tools
  python scripts/seed_registry.py --export tools.json # Export tool schemas
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mcp/servers.yaml",
        help="Path to MCP servers config YAML.",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export tool schemas to JSON file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured server categories and exit.",
    )
    return parser.parse_args()


async def discover_all(config_path: str) -> None:
    """Discover tools from all configured MCP servers.

    Args:
        config_path: Path to servers.yaml.
    """
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry(config_path=config_path)
    registry.load_from_config()

    print(f"  Configured servers: {list(registry._servers.keys())}")
    print()

    total = 0
    for category in registry._servers:
        print(f"  Spawning {category}...")
        tools = await registry.discover_tools(category)
        total += len(tools)
        print(f"    ✓ Discovered {len(tools)} tools")

        for tool in tools:
            destr = " [DESTRUCTIVE]" if tool.is_destructive else ""
            print(f"      • {tool.name} ({tool.category.value}){destr}")

    print()
    print(f"  Total: {total} tools across {len(registry._servers)} servers")
    return registry


def main() -> int:
    """Seed the tool registry from MCP servers.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()

    print("═══ ORION MCP Tool Discovery ═══")
    print()

    if args.list:
        import yaml

        path = Path(args.config)
        if not path.exists():
            print(f"  ✗ Config not found: {args.config}")
            return 1

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        servers = config.get("servers", {})
        print("  Configured MCP servers:")
        for cat, srv in servers.items():
            print(f"    • {cat}: {srv.get('command', '?')} {' '.join(srv.get('args', []))}")
        return 0

    try:
        registry = asyncio.run(discover_all(args.config))
    except Exception as exc:
        print(f"  ✗ Discovery failed: {exc}")
        return 1

    # Export if requested
    if args.export and registry:
        export_data = []
        for tool in registry.list_tools():
            export_data.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "is_destructive": tool.is_destructive,
                    "params_schema": tool.params_schema,
                    "server_category": tool.server_category,
                }
            )

        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"  ✓ Exported to {args.export}")

    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
