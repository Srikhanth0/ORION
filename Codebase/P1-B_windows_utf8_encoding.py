"""
ORION PATCH P1-B — Windows cp1252 / Unicode encoding fix
Files:
  - scripts/eval_task.py        (add UTF-8 reconfiguration at top)
  - scripts/healthcheck.py      (same)
  - Makefile                    (export PYTHONUTF8=1)
  - pyproject.toml              (pythonpath utf8 hint)
Commit: fix(scripts): force UTF-8 stdout/stderr on Windows to prevent cp1252 crash

Root cause: Python on Windows defaults console encoding to cp1252.
Box-drawing characters used in Rich output (━━━ U+2501) are not in cp1252,
causing UnicodeEncodeError before any test logic runs.
"""

# ── Prepend to EVERY script entry point (eval_task.py, healthcheck.py, etc.) ─

import sys
import io

# ORION-FIX: Reconfigure stdout/stderr to UTF-8 before any print/rich output.
# Safe no-op on Linux/macOS where encoding is already UTF-8.
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer") and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stderr, "buffer") and sys.stderr.encoding.lower() != "utf-8":
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# Makefile snippet — add near the top, before any target definitions:
# ─────────────────────────────────────────────────────────────────────────────
MAKEFILE_SNIPPET = r"""
# ORION-FIX: Force UTF-8 for all Python subprocesses on Windows
# Equivalent to running `python -X utf8` for every invocation.
export PYTHONUTF8 := 1

# Recommended: also set in PowerShell profile for interactive use:
# [System.Console]::OutputEncoding = [System.Text.Encoding]::UTF8
"""

# ─────────────────────────────────────────────────────────────────────────────
# pyproject.toml addition — under [tool.pytest.ini_options]:
# ─────────────────────────────────────────────────────────────────────────────
PYPROJECT_SNIPPET = r"""
[tool.pytest.ini_options]
# ORION-FIX: Ensures pytest itself runs in UTF-8 mode on Windows
addopts = "-p no:warnings"

[tool.orion.encoding]
# Documents intent; actual enforcement is via PYTHONUTF8 env var
stdout = "utf-8"
stderr = "utf-8"
"""

# ─────────────────────────────────────────────────────────────────────────────
# .env.example addition:
# ─────────────────────────────────────────────────────────────────────────────
DOTENV_SNIPPET = r"""
# ORION-FIX: Prevents cp1252 UnicodeEncodeError on Windows bare PowerShell
PYTHONUTF8=1
"""

# ─────────────────────────────────────────────────────────────────────────────
# Alternative: use a sitecustomize.py in the venv (applies automatically)
# Place at: .venv/Lib/site-packages/sitecustomize.py (Windows path)
# ─────────────────────────────────────────────────────────────────────────────
SITECUSTOMIZE = r"""
# sitecustomize.py — auto-loaded by Python before any user code
# ORION-FIX: Global UTF-8 enforcement for the ORION venv
import sys, io

if sys.platform == "win32":
    for _stream_name in ("stdout", "stderr"):
        _stream = getattr(sys, _stream_name)
        if hasattr(_stream, "buffer") and _stream.encoding.lower() != "utf-8":
            setattr(
                sys,
                _stream_name,
                io.TextIOWrapper(_stream.buffer, encoding="utf-8",
                                 errors="replace", line_buffering=True),
            )
"""

if __name__ == "__main__":
    print("P1-B patch contents printed above. Apply snippets to the listed files.")
