---
name: orion-debugger
description: >
  Diagnose, patch, and re-test the ORION agentic OS automation stack on Windows/WSL environments.
  Use this skill whenever a test run report is provided, or when the user mentions uvx/npx tool
  loading failures, OpenRouter 404s, Qdrant connectivity errors, Unicode/encoding crashes,
  or MCP tool registry returning 0 tools. Applies fixes across configs, providers, and
  environment bootstrap — then re-runs the full task suite to validate. Trigger on any
  ORION test report, break-point document, or "why is ORION failing" question.
---

# ORION Debugger Skill

A structured skill for diagnosing, patching, and re-validating ORION on Windows and WSL2.
Covers the full fix-cycle: triage → patch → verify → re-test.

---

## Phase 1 — Triage

Parse the incoming break-point report and classify every issue by severity tier:

| Tier | Label | Action |
|------|-------|--------|
| P0 | Blocks core functionality | Fix before anything else |
| P1 | Degrades reliability | Fix in same pass |
| P2 | Edge case / observability | Fix after P0+P1 confirmed |

Extract the following facts from the report:
- Python version, OS, shell (PowerShell vs WSL2)
- Which LLM providers are reachable / unreachable
- Whether Qdrant is running
- MCP tools loaded count
- Each task's failure mode (timeout / crash / wrong output)

---

## Phase 2 — Apply Patches

Work through patches in P0 → P1 → P2 order. For each patch:
1. Identify the exact file to change
2. Write the minimal diff (no unrelated refactors)
3. Add a comment `# ORION-FIX: <reason>` on the changed line
4. Record the patch in the session summary

### P0-A: Windows uvx / npx binary resolution

**Root cause**: `uvx` and `npx` are not on `PATH` in bare-metal PowerShell.
The tool registry calls them as bare subprocess commands which fail silently,
leaving `tools_loaded = 0`.

**Fix file**: `orion/tools/registry.py`

```python
# ORION-FIX: Resolve platform-correct binary names for uvx/npx on Windows
import shutil, sys

def _resolve_bin(name: str) -> str:
    """Return the correct executable name for the current platform."""
    if sys.platform == "win32":
        # On Windows, prefer the .cmd shim that lives in %APPDATA%\npm
        cmd_variant = name + ".cmd"
        if shutil.which(cmd_variant):
            return cmd_variant
    return name  # Linux / WSL2 / macOS — name is correct as-is

UVX_BIN = _resolve_bin("uvx")
NPX_BIN  = _resolve_bin("npx")
```

Then replace every bare `"uvx"` and `"npx"` string in subprocess calls
inside `registry.py` with `UVX_BIN` and `NPX_BIN`.

Also add a startup check in `orion/api/server.py`:

```python
# ORION-FIX: Warn early if uvx/npx are missing so failure is obvious
from orion.tools.registry import UVX_BIN, NPX_BIN
import shutil, logging

log = logging.getLogger(__name__)

for _bin in (UVX_BIN, NPX_BIN):
    if not shutil.which(_bin):
        log.warning(
            "Binary '%s' not found on PATH. "
            "MCP tools requiring it will fail to load. "
            "On Windows, run inside WSL2 or install Node.js + uv globally.", _bin
        )
```

### P0-B: OpenRouter model 404

**Root cause**: `mistralai/mistral-7b-instruct` was removed from OpenRouter's free tier.

**Fix file**: `configs/llm/openrouter.yaml`

```yaml
# ORION-FIX: Updated to active free-tier model (mistral-7b removed 2026-Q1)
provider: openrouter
models:
  default: google/gemma-2-9b-it:free      # primary free fallback
  reasoning: mistralai/mixtral-8x7b-instruct:nitro  # paid, optional
  fast:    meta-llama/llama-3.1-8b-instruct:free

base_url: https://openrouter.ai/api/v1
timeout: 30
retry_on_status: [429, 500, 502, 503]
```

**Fix file**: `.env.example` — update the comment:

```bash
# ORION-FIX: Free models change frequently. Verify at https://openrouter.ai/models?q=free
OPENROUTER_DEFAULT_MODEL=google/gemma-2-9b-it:free
```

### P1-A: Qdrant optional dependency

**Root cause**: When Qdrant is not running, the Verifier crashes on any task that
touches long-term memory because `longterm.py` raises on connection failure with
no fallback.

**Fix file**: `orion/memory/longterm.py`

```python
# ORION-FIX: Graceful degradation when Qdrant is unavailable
import logging
from typing import Optional

log = logging.getLogger(__name__)

class LongTermMemory:
    def __init__(self, url: str, collection: str):
        self._client = None
        self._available = False
        try:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(url=url, timeout=3)
            self._client.get_collections()   # health-check
            self._available = True
            log.info("Long-term memory: Qdrant connected at %s", url)
        except Exception as exc:
            log.warning(
                "Long-term memory: Qdrant unavailable (%s). "
                "Falling back to in-process dict store — "
                "run `podman-compose up qdrant` for persistence.", exc
            )
            self._fallback: dict[str, list] = {}

    def store(self, key: str, vector: list[float], payload: dict) -> None:
        if self._available:
            # ... existing qdrant upsert logic ...
            pass
        else:
            self._fallback.setdefault(key, []).append(payload)

    def search(self, vector: list[float], top_k: int = 5) -> list[dict]:
        if self._available:
            # ... existing qdrant query logic ...
            return []
        # naive fallback: return last top_k stored items
        all_items = [p for v in self._fallback.values() for p in v]
        return all_items[-top_k:]
```

**Fix file**: `orion/api/server.py` — remove Qdrant from the readiness gate:

```python
# ORION-FIX: Qdrant is now optional; readiness only fails if LLM is unreachable
@app.get("/ready")
async def readiness():
    llm_ok = await llm_router.health_check()
    return {"status": "ok" if llm_ok else "degraded", "qdrant": memory.is_available()}
```

### P1-B: Windows Unicode / cp1252 crash

**Root cause**: Python on Windows defaults to `cp1252` console encoding.
The test harness prints box-drawing characters (`━━━`) which are not in cp1252.

**Fix file**: `scripts/eval_task.py` (and any script using `print`)

```python
# ORION-FIX: Force UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
import sys, io

if sys.platform == "win32" and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
```

**Fix file**: `Makefile` — add `PYTHONUTF8=1` to all `python` invocations:

```makefile
# ORION-FIX: Force UTF-8 for all Python subprocesses on Windows
export PYTHONUTF8 := 1
```

**Fix file**: `pyproject.toml` — add console_scripts entry point with UTF-8 flag:

```toml
[tool.orion]
# ORION-FIX: Ensure UTF-8 globally
python_utf8 = true
```

### P2: SSE subtask stream polling

**Root cause**: The test harness was calling `/v1/tasks/{id}` (polling)
instead of `/v1/tasks/{id}/stream` (SSE). Intermediate subtask steps
were invisible, making timeout diagnosis impossible.

**Fix file**: `tests/integration/test_pipeline.py`

```python
# ORION-FIX: Use SSE stream endpoint to capture granular subtask steps
import httpx, json

async def stream_task(task_id: str, base_url: str) -> list[dict]:
    """Collect all SSE events for a task, yielding each step dict."""
    events = []
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("GET", f"{base_url}/v1/tasks/{task_id}/stream") as r:
            async for line in r.aiter_lines():
                if line.startswith("data:"):
                    payload = json.loads(line[5:].strip())
                    events.append(payload)
                    if payload.get("status") in ("completed", "failed"):
                        break
    return events
```

---

## Phase 3 — Re-Test Protocol

After all patches are applied, run the three canonical tasks in order.
For each task capture: status, duration, LLM path used, tools invoked, and
any remaining errors.

```bash
# Set UTF-8 before anything else
export PYTHONUTF8=1

# Verify tool registry loads correctly now
python -c "from orion.tools.registry import build_registry; r = build_registry(); print(f'Tools loaded: {len(r)}')"

# Run the eval tasks
python scripts/eval_task.py --task shell_basic     --stream --timeout 60
python scripts/eval_task.py --task reasoning_mem   --stream --timeout 90
python scripts/eval_task.py --task gui_vision      --stream --timeout 90
```

Expected outcomes after patches:
- Task 1 (Shell): PASS — uvx/npx resolve, tools > 0
- Task 2 (Reasoning + Memory): PASS with degraded memory warning (Qdrant offline → fallback dict)
- Task 3 (GUI Vision): PASS if ngrok tunnel is up; SKIP-with-note if tunnel is down

---

## Phase 4 — Report

After re-testing, produce a structured report:

```
# ORION Re-Test Report
Date: <date>
Patches applied: P0-A, P0-B, P1-A, P1-B, P2

## Results
| Task | v1.0.0 | v1.0.1-fix | Notes |
|------|--------|------------|-------|
| Shell     | FAIL(timeout) | PASS | tools_loaded now > 0 |
| Reasoning | FAIL(timeout) | PASS(degraded) | Qdrant fallback active |
| GUI Vision| FAIL(timeout) | PASS | ngrok context mapped |

## Remaining risks
- Long-term memory is ephemeral without Qdrant (in-process fallback)
- OpenRouter free models change frequently — add model-health CI check
```
