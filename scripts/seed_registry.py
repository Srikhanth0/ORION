#!/usr/bin/env python3
"""ORION seed_registry — pre-populate the tool registry from Composio.

Authenticates with Composio and loads all enabled app tools into
the ORION tool registry for inspection and testing.

Usage:
    python scripts/seed_registry.py [--apps github shell]
    python scripts/seed_registry.py --list
    python scripts/seed_registry.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Pre-populate the ORION tool registry from Composio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/seed_registry.py                    # Load all enabled apps
  python scripts/seed_registry.py --apps github shell # Load specific apps
  python scripts/seed_registry.py --list              # List available apps
        """,
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        default=None,
        help="Specific Composio apps to load (default: all from config).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available Composio apps and exit.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mcp/composio.yaml",
        help="Path to Composio config YAML.",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export tool schemas to JSON file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load Composio config from YAML.

    Args:
        config_path: Path to YAML config.

    Returns:
        Parsed config dict.
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        print(f"  ✗ Config not found: {config_path}")
        return {}

    with open(path) as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    """Seed the tool registry.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()

    print("═══ ORION Tool Registry Seeder ═══")
    print()

    # Load config
    config = load_config(args.config)
    composio_config = config.get("composio", {})

    # Resolve API key
    api_key_env = composio_config.get(
        "api_key_env", "COMPOSIO_API_KEY"
    )
    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(
            f"  ⚠ No API key found in ${api_key_env}"
        )
        print("  Proceeding without authentication...")
        print()

    # Determine apps to load
    enabled_apps = args.apps or composio_config.get("enabled_apps", [])

    print(f"  Apps to load: {enabled_apps or 'all'}")
    print()

    # Initialize registry
    from orion.tools.registry import ToolRegistry

    registry = ToolRegistry(api_key=api_key)

    if args.list:
        print("  Available Composio apps:")
        for app in enabled_apps:
            print(f"    • {app}")
        return 0

    # Load tools
    print("  Loading tools from Composio...")
    registry.load(enabled_apps=enabled_apps or None)

    tool_count = registry.tool_count
    print(f"  ✓ Loaded {tool_count} tools")
    print()

    # Display loaded tools
    if tool_count > 0:
        print("  Registered tools:")
        for tool in registry.list_tools():
            destr = " [DESTRUCTIVE]" if tool.is_destructive else ""
            print(
                f"    • {tool.name} ({tool.category.value})"
                f"{destr}"
            )
        print()

    # Export if requested
    if args.export:
        export_data = []
        for tool in registry.list_tools():
            export_data.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "is_destructive": tool.is_destructive,
                "params_schema": tool.params_schema,
            })

        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"  ✓ Exported to {args.export}")

    print("  Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
