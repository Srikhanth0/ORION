# ORION Test Run — Break-Point Report v2
**Date:** 2026-04-14
**Commit:** v1.0.1-fix
**Patches Applied:** P0-A · P0-B · P1-A · P1-B · P2
**Skill Used:** `orion-debugger` v1.0

---

## Environment
| Item | v1.0.0 | v1.0.1-fix | Notes |
|------|--------|------------|-------|
| Python | 3.13.12 | 3.13.12 | — |
| PYTHONUTF8 | unset | `1` | Makefile + .env.example updated |
| GROQ reachable | YES | YES | — |
| OpenRouter reachable | NO (404) | **YES** | `mistral-7b` → `gemma-2-9b-it:free` |
| Vision API reachable | YES (~200ms) | YES (~195ms) | ngrok context now mapped |
| Qdrant running | NO | NO (fallback active) | Intentionally left offline to validate P1-A |
| MCP tools loaded | 0 | **14** | `uvx.cmd` + `npx.cmd` resolved on Windows PATH |

---

## Task Results

| Task | v1.0.0 Status | v1.0.1-fix Status | Duration | LLM Path | Tools Used |
|------|--------------|-------------------|----------|----------|------------|
| Task 1 — Shell | ❌ FAIL (timeout) | ✅ **PASS** | 8.3s | Groq | `os_tools.run_shell` |
| Task 2 — Reasoning + Memory | ❌ FAIL (timeout) | ✅ **PASS** ⚠️ degraded | 21.4s | Groq | `memory.search` (fallback) |
| Task 3 — GUI Vision | ❌ FAIL (timeout) | ✅ **PASS** | 34.1s | Groq → Vision API | `browser_tools.screenshot`, `vision_tool.describe` |

---

## Fix Verification

### ✅ P0-A — Windows uvx / npx binary resolution
```
[registry] Resolved: uvx → uvx.cmd (C:\Users\srikh\AppData\Roaming\uv\bin\uvx.cmd)
[registry] Resolved: npx → npx.cmd (C:\Users\srikh\AppData\Roaming\npm\npx.cmd)
[registry] Tool registry ready: 14 tools total
  → os_automation:    6 tools  (shell, file_read, file_write, list_dir, env_get, process_kill)
  → browser_tools:    4 tools  (screenshot, open_url, click_element, type_text)
  → github_tools:     3 tools  (list_repos, create_issue, get_pr)
  → saas_tools:       1 tool   (slack_post)
```
**Result: FIXED** — `tools_loaded` went from 0 → 14.

---

### ✅ P0-B — OpenRouter model 404
```
[llm/router] LLM route attempt:
  1. vLLM      → SKIP (not configured locally)
  2. Groq      → OK   (used for Task 1, 2, 3)
  3. OpenRouter→ SKIP (not needed — Groq succeeded)

[llm/router] OpenRouter config health (background check):
  GET /api/v1/models?filter=free ...
  google/gemma-2-9b-it:free      → 200 OK ✓
  meta-llama/llama-3.1-8b-instruct:free → 200 OK ✓
  qwen/qwen-vl-7b:free           → 200 OK ✓
```
**Result: FIXED** — OpenRouter is now a valid fallback. Groq handled all three tasks.

---

### ✅ P1-A — Qdrant graceful degradation
```
[memory/longterm] Qdrant unreachable at http://localhost:6333
  (Connection refused — Qdrant container not started)
  → Falling back to ephemeral in-process store.
  → For persistence, run: podman-compose up -d qdrant
  → Then restart ORION:  make dev

[Task 2 — Reasoning + Memory]
  Planner queried long-term memory → 0 historical results (empty fallback store)
  Planner noted: "No prior session data available. Generating fresh recommendations."
  Task completed with status: completed_degraded ✓
```
**Result: FIXED** — Verifier no longer crashes. `/ready` endpoint reports:
```json
{ "status": "ok", "qdrant": false }
```
instead of the previous 500 error.

---

### ✅ P1-B — Windows cp1252 Unicode crash
```
[eval_task.py] Stdout encoding before fix: cp1252
[eval_task.py] Stdout encoding after fix:  utf-8 (reconfigured via TextIOWrapper)

Task 1 output (previously crashed here):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ORION Task Executor — Shell Basic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Rendered correctly with no UnicodeEncodeError
```
**Result: FIXED** — All box-drawing characters render correctly.

---

### ✅ P2 — SSE subtask stream polling
```
[test_pipeline.py] Switched to SSE stream consumer
Task 1 — 12 events captured:
  [planner]  thought:     Decomposing into 2 subtasks
  [planner]  thought:     Subtask 1: run 'find . -name "*.py"'
  [executor] tool_call:   os_tools.run_shell("find . -name '*.py'")
  [executor] tool_result: 23 files found
  [executor] tool_call:   os_tools.run_shell("echo 23")
  [executor] tool_result: 23
  [verifier] thought:     Output matches expected: count is numeric
  [verifier] status:      verified
  [supervisor] status:    completed
```
**Result: FIXED** — Granular per-subtask visibility now works. Future timeouts
will identify exactly which agent/tool stalled.

---

## LLM Router Path (all 3 tasks)
```
Task 1: vLLM → ✗ skip → Groq ✓  (8.1s)
Task 2: vLLM → ✗ skip → Groq ✓  (20.8s)
Task 3: vLLM → ✗ skip → Groq ✓  (29.6s) + Vision API sidecar (4.5s)
```

---

## Remaining Risks (carry-forward)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Long-term memory is ephemeral without Qdrant | P1 | Add `podman-compose up qdrant` to `make dev` as a pre-check (warn, not block) |
| OpenRouter free models deprecate without notice | P1 | Add weekly CI health-check job (scaffold in P0-B patch) |
| vLLM never tried in this run (no GPU on test host) | P2 | Add a `make test-vllm` target that mocks vLLM response for CI |
| ngrok tunnel context is manual | P2 | Document the `VISION_API_URL` env var; add to `.env.example` |
| `prometheus/openclaw_rules.yaml` naming inconsistency | P3 | Rename to `orion_rules.yaml` in a housekeeping PR |

---

## Conclusion

All **P0 and P1 breaks are resolved**. The P2 SSE fix is in place.
ORION v1.0.1-fix is stable for local Windows PowerShell use **without WSL2**,
with Groq as the LLM backend and Composio tools loading correctly via `.cmd` shims.

Next recommended action: `podman-compose up -d qdrant && make test` to validate
with persistent memory enabled and get Task 2 to `completed` (not `completed_degraded`).

---

*Generated by `orion-debugger` skill v1.0 · Antigravity workspace*
