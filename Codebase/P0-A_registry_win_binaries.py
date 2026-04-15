"""
ORION PATCH P0-A — Windows uvx / npx binary resolution
File: orion/tools/registry.py  (prepend this block at the top, then replace bare "uvx"/"npx" strings)
Commit: fix(tools): resolve platform-correct binary names for uvx/npx on Windows
"""

import shutil
import sys
import logging
import subprocess
from typing import Optional

log = logging.getLogger(__name__)


# ── Platform binary resolution ────────────────────────────────────────────────

def _resolve_bin(name: str) -> str:
    """
    Return the correct executable name for the current platform.

    On Windows (bare PowerShell), Node packages install .cmd shims into
    %APPDATA%\\npm — e.g. npx.cmd — which are NOT on PATH as bare 'npx'.
    uv installs uvx.exe into %LOCALAPPDATA%\\uv\\bin.

    Inside WSL2, names are identical to Linux.
    """
    if sys.platform == "win32":
        # Try .cmd shim first (npm-installed tools: npx, yarn, pnpm …)
        cmd_shim = name + ".cmd"
        if shutil.which(cmd_shim):
            return cmd_shim
        # Try .exe directly (uv-installed tools: uvx.exe …)
        exe_shim = name + ".exe"
        if shutil.which(exe_shim):
            return exe_shim
    return name  # Linux / WSL2 / macOS: names are correct as-is


# Resolved binary names — use these everywhere instead of bare string literals
UVX_BIN = _resolve_bin("uvx")
NPX_BIN  = _resolve_bin("npx")


def _warn_missing(bin_name: str) -> None:
    """Emit an actionable warning when a required binary is absent."""
    if not shutil.which(bin_name):
        log.warning(
            "Binary '%s' not found on PATH. MCP tools that depend on it will "
            "fail to load.\n"
            "  • Windows bare PowerShell: install Node.js LTS + uv, then restart shell\n"
            "  • Recommended: run ORION inside WSL2 to avoid Windows PATH fragmentation\n"
            "  • Quick check: run `where %s` (PowerShell) or `which %s` (bash)",
            bin_name, bin_name, bin_name,
        )


# Validate at import time so the failure is loud and early
_warn_missing(UVX_BIN)
_warn_missing(NPX_BIN)


# ── Drop-in replacement helpers ───────────────────────────────────────────────

def run_uvx(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a uvx command using the platform-correct binary."""
    return subprocess.run([UVX_BIN, *args], **kwargs)


def run_npx(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run an npx command using the platform-correct binary."""
    return subprocess.run([NPX_BIN, *args], **kwargs)


# ── Registry builder (existing logic — wire in the helpers above) ─────────────

def build_registry() -> dict:
    """
    Scan all enabled MCP configs and load tools.
    Returns a dict of {tool_name: tool_schema}.

    ORION-FIX: Uses run_uvx / run_npx instead of bare subprocess calls so
    Windows .cmd shims are correctly resolved.
    """
    tools: dict = {}

    # Example: load the os_automation MCP via uvx
    try:
        result = run_uvx(
            ["mcp-server-os-automation", "--list-tools"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            import json
            loaded = json.loads(result.stdout)
            tools.update(loaded)
            log.info("Loaded %d OS automation tools via uvx", len(loaded))
        else:
            log.warning("uvx os-automation tool list failed: %s", result.stderr)
    except FileNotFoundError:
        log.error("uvx binary '%s' not found — OS automation tools unavailable", UVX_BIN)
    except Exception as exc:
        log.error("Failed to load OS tools: %s", exc)

    # Example: load browser tools via npx
    try:
        result = run_npx(
            ["@modelcontextprotocol/server-puppeteer", "--list-tools"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            import json
            loaded = json.loads(result.stdout)
            tools.update(loaded)
            log.info("Loaded %d browser tools via npx", len(loaded))
        else:
            log.warning("npx browser tool list failed: %s", result.stderr)
    except FileNotFoundError:
        log.error("npx binary '%s' not found — browser tools unavailable", NPX_BIN)
    except Exception as exc:
        log.error("Failed to load browser tools: %s", exc)

    log.info("Tool registry ready: %d tools total", len(tools))
    return tools
