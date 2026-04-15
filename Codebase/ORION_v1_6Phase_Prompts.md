# ORION v1.0 — 6-Phase Implementation Prompts
> Feed each phase into your AI coding assistant (Cursor Composer, Aider, Claude Code) sequentially.
> Complete all tests before moving to the next phase.

---

## PHASE 1 — Foundation & Podman Migration
**Goal:** Replace Docker with Podman, swap Qdrant with ChromaDB, set up CLI entry point.

```markdown
# ORION v1.0 — Phase 1: Foundation & Podman Migration

You are upgrading the ORION OS Automation Agent from v0.1 to v1.0.
Work through these steps exactly in order. Do not skip steps. After each file change, run the tests described.

## Objectives
1. Replace all Docker references with Podman (rootless, daemonless).
2. Replace Qdrant long-term memory with embedded ChromaDB (zero external dependency).
3. Create the terminal CLI entry point `orion_cli.py` using `rich` + `prompt_toolkit`.
4. Update `pyproject.toml` dependencies accordingly.

## Step 1 — Podman Compose File
- Rename `docker-compose.yml` → `podman-compose.yml`.
- Replace `image: qdrant/qdrant` service with nothing (ChromaDB is embedded, no container needed).
- Replace any `build: .` + Dockerfile references with Podman-compatible equivalents.
- Update `Makefile` targets: `docker-up` → `podman-up`, `docker-down` → `podman-down`.
- Add `make podman-build` that runs `podman build -t orion:latest .`.
- Replace `Dockerfile` base image with `python:3.13-slim`. Ensure no Docker-specific syntax remains.

## Step 2 — ChromaDB Long-Term Memory
Navigate to `orion/memory/longterm.py`.
- Remove ALL Qdrant imports and client instantiation.
- Implement class `LocalLongTermMemory` backed by `chromadb.PersistentClient(path=".orion_memory")`.
- Collection name: `"orion_tasks"`.
- Method `store(task_id: str, summary: str, embedding_text: str)` → upsert to collection.
- Method `retrieve(query: str, n_results: int = 5) → list[dict]` → query collection, return metadata list.
- Method `clear()` → delete and recreate collection.
- Update `configs/memory/longterm.yaml` to remove Qdrant host/port fields; add `persist_path: ".orion_memory"`.
- Update `orion/api/routes/status.py` readiness probe: remove Qdrant health check, replace with `chromadb.PersistentClient` instantiation check.
- Update `tests/unit/test_longterm_memory.py` to mock ChromaDB calls instead of Qdrant HTTP calls.
- Update `tests/integration/test_memory.py` to use an in-memory ChromaDB client for speed.

## Step 3 — Terminal CLI
Create `orion_cli.py` at project root.
- Use `rich` for formatting and `prompt_toolkit` for the REPL loop.
- On startup: print ORION v1.0 ASCII banner using `rich.panel.Panel`.
- REPL loop: prompt `[ORION] › ` → read input → POST to `http://localhost:8080/v1/tasks` with `{"instruction": input}` → print `task_id`.
- Command `/stream <task_id>` → open SSE stream at `GET /v1/tasks/{id}/stream` and print each event live using `rich.live.Live`.
- Command `/tools` → GET `/v1/tools` and pretty-print with `rich.table.Table`.
- Command `/quit` → exit.
- Command `/help` → print command list.
- Add entry point to `pyproject.toml`: `[project.scripts] orion = "orion_cli:main"`.

## Step 4 — Dependency Updates
In `pyproject.toml`:
- Remove: `qdrant-client`
- Add: `chromadb>=0.5`, `rich>=13`, `prompt-toolkit>=3.0`
- Ensure `uv lock` is run after changes.

## Validation
Run: `make test`
All 176 tests must pass (update any that referenced Qdrant).
Run: `uv run orion` — CLI must start without errors.
```

---

## PHASE 2 — Open-Source MCP Migration
**Goal:** Rip out Composio entirely. Implement native MCP stdio client using the `mcp` Python SDK.

```markdown
# ORION v1.0 — Phase 2: Open-Source MCP Migration

## Objectives
Remove the Composio cloud SDK and replace every tool call with the official open-source
Anthropic MCP Python SDK communicating with local MCP servers over stdio.

## Step 1 — Remove Composio
- Delete `configs/mcp/composio.yaml`.
- Remove `composio-core`, `composio-agentscope` (or equivalent) from `pyproject.toml`.
- Remove all `import composio` statements from `orion/tools/mcp_client.py` and `orion/tools/registry.py`.
- Remove `tests/integration/test_composio_mcp.py` entirely.

## Step 2 — MCP Client Rewrite
Rewrite `orion/tools/mcp_client.py`:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from orion.core.result import Result

class MCPStdioClient:
    """Manages a persistent stdio session to a local MCP server process."""

    def __init__(self, server_params: StdioServerParameters):
        self._params = server_params
        self._session: ClientSession | None = None

    async def __aenter__(self):
        self._streams = await asyncio.get_event_loop().run_in_executor(
            None, lambda: stdio_client(self._params).__enter__()
        )
        read, write = self._streams
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.__aexit__(*args)

    async def call_tool(self, tool_name: str, arguments: dict) -> Result:
        if not self._session:
            return Result.failure("MCP session not initialized")
        try:
            result = await self._session.call_tool(tool_name, arguments)
            return Result.success(result.content)
        except Exception as e:
            return Result.failure(str(e))

    async def list_tools(self) -> list[dict]:
        if not self._session:
            return []
        response = await self._session.list_tools()
        return [{"name": t.name, "description": t.description, "schema": t.inputSchema}
                for t in response.tools]
```

## Step 3 — Tool Registry Rewrite
Rewrite `orion/tools/registry.py` to spawn local MCP server processes:

- `os_tools` → spawn: `npx -y @modelcontextprotocol/server-filesystem /`
  (StdioServerParameters command="npx", args=["-y","@modelcontextprotocol/server-filesystem","/"])
- `browser_tools` → spawn: `npx -y @playwright/mcp@latest`
- `github_tools` → spawn: `npx -y @modelcontextprotocol/server-github`
  (requires env var GITHUB_PERSONAL_ACCESS_TOKEN)
- `saas_tools` → spawn: `python -m orion.tools.local_saas_server` (a simple local stub)

Each entry in the registry is a `dict` with keys: `category`, `server_params`, `client: MCPStdioClient | None`.

Method `get_client(category: str) -> MCPStdioClient` — lazy-initializes and caches the client.
Method `list_all_tools() -> list[dict]` — iterates all categories, calls `client.list_tools()`.
Method `call(category: str, tool_name: str, arguments: dict) -> Result` — delegates to client.

## Step 4 — Executor Integration
In `orion/agents/executor.py`:
- Replace any Composio action calls with `registry.call(category, tool_name, args)`.
- Import `MCPStdioClient` and `ToolRegistry`.
- Ensure every tool call is `await`ed (all async).

## Step 5 — Config
Create `configs/mcp/servers.yaml`:
```yaml
servers:
  os_tools:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/"]
    env: {}
  browser_tools:
    command: npx
    args: ["-y", "@playwright/mcp@latest"]
    env: {}
  github_tools:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_PAT}"
```

## Step 6 — Environment Variables
Add to `.env.example`:
```
GITHUB_PAT=your_github_personal_access_token
MCP_TIMEOUT_SECONDS=30
```

## Validation
- `make test` — all tests pass (mock stdio in unit tests using `unittest.mock.AsyncMock`).
- `curl -X GET http://localhost:8080/v1/tools` — returns 28+ tools with no Composio references.
- `curl -X POST http://localhost:8080/v1/tasks -d '{"instruction":"list files in /tmp"}'` — succeeds end-to-end.
```

---

## PHASE 3 — AgentScope Integration & Async Pipeline
**Goal:** Migrate all 4 agents to AgentScope. Implement DAG-based parallel execution.

```markdown
# ORION v1.0 — Phase 3: AgentScope + Async Pipeline

## Objectives
1. Rewrite all 4 HiClaw agents using AgentScope's `agentscope.agents.AgentBase`.
2. Upgrade Planner to output a dependency DAG with `depends_on` fields.
3. Rewrite `dispatcher.py` to execute independent subtasks with `asyncio.gather`.

## Step 1 — AgentScope Setup
Add to `pyproject.toml`: `agentscope>=0.1.0`
Create `orion/agentscope_config.py`:
```python
import agentscope
from pathlib import Path
import yaml

def init_agentscope(router_config_path: str = "configs/llm/router.yaml"):
    cfg = yaml.safe_load(Path(router_config_path).read_text())
    # Map ORION router config to AgentScope model configs
    model_configs = []
    for provider in cfg.get("providers", []):
        model_configs.append({
            "model_type": provider["type"],  # "openai_chat" or "post_api_chat"
            "config_name": provider["name"],
            "model_name": provider["model"],
            "api_key": provider.get("api_key", ""),
            "client_args": provider.get("client_args", {}),
        })
    agentscope.init(model_configs=model_configs, project="ORION")
```

## Step 2 — Agent Rewrites
For each agent in `orion/agents/`, rewrite to extend `agentscope.agents.AgentBase`:

**Planner (`orion/agents/planner.py`)**:
```python
from agentscope.agents import AgentBase
from agentscope.message import Msg
import json

class PlannerAgent(AgentBase):
    name = "Planner"
    
    def reply(self, x: Msg) -> Msg:
        # Load Jinja2 system prompt from prompts/planner_system.j2
        # Retrieve from working + long-term memory
        # Call model to generate subtask DAG as JSON
        # Return Msg with content = {"subtasks": [...], each with "id", "action", "tool", "depends_on": []}
        ...
```

**Executor (`orion/agents/executor.py`)**:
```python
class ExecutorAgent(AgentBase):
    name = "Executor"
    # Receives subtask list from Planner
    # Passes each through safety gate before tool call
    # Accepts optional screenshot_data bytes for vision subtasks
    ...
```

**Verifier (`orion/agents/verifier.py`)**:
```python
class VerifierAgent(AgentBase):
    name = "Verifier"
    # Checks executor output against expected result
    # Returns pass/fail + reason
    ...
```

**Supervisor (`orion/agents/supervisor.py`)**:
```python
class SupervisorAgent(AgentBase):
    name = "Supervisor"
    # Decides: retry(→Planner) | partial_complete | complete
    # On retry: attaches failure context to next Planner Msg
    ...
```

## Step 3 — DAG Dispatcher
Rewrite `orion/orchestrator/dispatcher.py`:

```python
import asyncio
from collections import defaultdict

async def execute_dag(subtasks: list[dict], executor: ExecutorAgent) -> dict:
    """
    Topological execution: run subtasks whose depends_on are all resolved,
    using asyncio.gather for independent nodes.
    """
    results = {}
    pending = {t["id"]: t for t in subtasks}
    
    while pending:
        # Find all tasks whose dependencies are satisfied
        ready = [
            t for t in pending.values()
            if all(dep in results for dep in t.get("depends_on", []))
        ]
        if not ready:
            raise RuntimeError("DAG cycle detected or unsatisfiable dependency")
        
        # Execute ready tasks in parallel
        coros = [executor.execute_subtask(t, results) for t in ready]
        batch_results = await asyncio.gather(*coros, return_exceptions=True)
        
        for task, result in zip(ready, batch_results):
            results[task["id"]] = result
            del pending[task["id"]]
    
    return results
```

## Step 4 — Planner Prompt Update
Update `prompts/planner_system.j2`:
Add instruction: "Output a JSON array of subtasks. Each subtask must have:
- id (string, unique)
- action (natural language description)
- tool_category (os_tools | browser_tools | github_tools | saas_tools | vision_tools)
- tool_name (specific MCP tool name)
- arguments (dict)
- depends_on (list of subtask ids that must complete before this one)"

## Validation
- `make test`
- Submit a multi-step task and verify subtasks execute in parallel where possible.
  Log output should show concurrent task IDs running simultaneously.
- `make eval` — all 10 canonical tasks must pass.
```

---

## PHASE 4 — Vision Layer (Qwen2.5-VL via Colab + ngrok)
**Goal:** Add screen-aware computer control. Tunnel a Colab-hosted vision model via ngrok.

```markdown
# ORION v1.0 — Phase 4: Vision & Computer Control

## Objectives
1. Create a Colab notebook that hosts Qwen2.5-VL and exposes it via ngrok.
2. Add `vision_tools` category to the MCP registry with screenshot + OCR + click tools.
3. Integrate pyautogui for translating bounding boxes to mouse clicks.

## Step 1 — Colab Vision Server
Create `scripts/colab_vision_server.ipynb` with these cells:

**Cell 1 — Install**:
```
!pip install transformers accelerate torch torchvision pyngrok flask pillow
```

**Cell 2 — Load model**:
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("Model loaded.")
```

**Cell 3 — Flask API**:
```python
from flask import Flask, request, jsonify
import base64, io
from PIL import Image

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    img_b64 = data["image_base64"]
    prompt = data.get("prompt", "Describe this screen. List all clickable UI elements with their approximate pixel coordinates as bounding boxes.")
    
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)
    
    output = processor.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"result": output})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": model_name})
```

**Cell 4 — Start with ngrok**:
```python
from pyngrok import ngrok
import threading, os

# Set your ngrok authtoken (get free at ngrok.com)
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
public_url = ngrok.connect(5000).public_url
print(f"Vision API URL: {public_url}")
print(f"Add to .env: VISION_API_URL={public_url}")

threading.Thread(target=lambda: app.run(port=5000, debug=False)).start()
```

## Step 2 — Vision Tool Implementation
Create `orion/tools/categories/vision_tools.py`:

```python
import asyncio, base64, io, os
import httpx
import pyautogui
from PIL import Image, ImageGrab

VISION_API_URL = os.getenv("VISION_API_URL", "")

async def take_screenshot() -> bytes:
    """Capture full screen and return PNG bytes."""
    screenshot = ImageGrab.grab()
    buf = io.BytesIO()
    screenshot.save(buf, format="PNG")
    return buf.getvalue()

async def analyze_screen(prompt: str = None) -> dict:
    """Take screenshot, send to Qwen2.5-VL, return structured result."""
    if not VISION_API_URL:
        raise RuntimeError("VISION_API_URL not set. Start Colab server and set env var.")
    
    img_bytes = await take_screenshot()
    img_b64 = base64.b64encode(img_bytes).decode()
    
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{VISION_API_URL}/analyze",
            json={"image_base64": img_b64, "prompt": prompt or ""}
        )
        resp.raise_for_status()
        return resp.json()

async def click_element(description: str) -> dict:
    """Ask vision model for coordinates of element, then click it."""
    result = await analyze_screen(
        prompt=f"Find the UI element matching: '{description}'. "
               "Return ONLY a JSON object: {{\"x\": <pixel_x>, \"y\": <pixel_y>, \"found\": true/false}}"
    )
    # Parse coordinates from model output
    import json, re
    text = result.get("result", "")
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if not match:
        return {"success": False, "error": "No coordinates found in model response"}
    coords = json.loads(match.group())
    if not coords.get("found"):
        return {"success": False, "error": f"Element '{description}' not found on screen"}
    pyautogui.click(coords["x"], coords["y"])
    return {"success": True, "clicked_at": {"x": coords["x"], "y": coords["y"]}}

async def type_text(text: str) -> dict:
    """Type text at current cursor position."""
    pyautogui.write(text, interval=0.05)
    return {"success": True, "typed": text}

async def press_key(key: str) -> dict:
    """Press a keyboard key (e.g. 'enter', 'tab', 'ctrl+c')."""
    pyautogui.hotkey(*key.split("+"))
    return {"success": True, "key": key}
```

## Step 3 — Registry Integration
In `orion/tools/registry.py`, add vision_tools category:
```python
"vision_tools": {
    "category": "vision_tools",
    "server_params": None,  # Not MCP stdio — direct Python calls
    "tools": [
        {"name": "take_screenshot", "fn": vision_tools.take_screenshot},
        {"name": "analyze_screen", "fn": vision_tools.analyze_screen},
        {"name": "click_element", "fn": vision_tools.click_element},
        {"name": "type_text", "fn": vision_tools.type_text},
        {"name": "press_key", "fn": vision_tools.press_key},
    ]
}
```

## Step 4 — Safety Gate Update
In `orion/safety/gate.py`, add vision tool rules:
```yaml
# in configs/safety/permissions.yaml
vision_tools:
  allowed: true
  require_approval_for: ["click_element", "type_text", "press_key"]
  reason: "Computer control actions require human approval in strict mode"
```

## Step 5 — Executor Vision Hook
In `orion/agents/executor.py`, add:
```python
async def execute_subtask(self, subtask: dict, prior_results: dict) -> dict:
    if subtask["tool_category"] == "vision_tools":
        # Route to direct vision_tools Python calls
        fn = self.registry.get_vision_tool(subtask["tool_name"])
        return await fn(**subtask.get("arguments", {}))
    else:
        # Route to MCP stdio client
        return await self.registry.call(
            subtask["tool_category"], subtask["tool_name"], subtask.get("arguments", {})
        )
```

## Step 6 — Environment
Add to `.env.example`:
```
VISION_API_URL=https://xxxx.ngrok.io   # From Colab cell 4 output
PYAUTOGUI_FAILSAFE=true
```

## Validation
- Start Colab notebook, copy ngrok URL to `.env`.
- `curl -X POST http://localhost:8080/v1/tasks -d '{"instruction":"Take a screenshot and describe what is on screen"}'`
- Check SSE stream for vision analysis output.
- `make test` — all tests pass (mock httpx in unit tests).
```

---

## PHASE 5 — Hardening, SSE Streaming & Eval Suite
**Goal:** Production-grade error handling, complete SSE pipeline, and full eval coverage.

```markdown
# ORION v1.0 — Phase 5: Hardening & Eval Suite

## Objectives
1. Complete the SSE real-time stream for every agent thought and tool call.
2. Add circuit breakers to LLM router and MCP clients.
3. Expand eval suite to 25 canonical tasks covering all tool categories.

## Step 1 — SSE Stream Completion
In `orion/api/routes/tasks.py`, the `GET /v1/tasks/{id}/stream` endpoint must emit:
- `event: planner_start` — when Planner begins
- `event: subtask_queued` + `data: {"id":..., "action":...}` — for each subtask
- `event: subtask_start` + `data: {"id":..., "tool":...}` — when Executor picks up a subtask
- `event: subtask_result` + `data: {"id":..., "success":..., "output":...}` — after each subtask
- `event: verifier_result` + `data: {"pass":..., "reason":...}`
- `event: supervisor_decision` + `data: {"decision": "complete"|"retry"|"partial"}`
- `event: done` + `data: {"task_id":..., "status":...}` — final event

Use `asyncio.Queue` per task_id. Agents push events to queue; SSE endpoint consumes.
Store queues in a module-level dict: `_task_queues: dict[str, asyncio.Queue] = {}`.

## Step 2 — Circuit Breakers
In `orion/llm/router.py`:
- Track consecutive failures per provider in a dict.
- After 3 consecutive failures: mark provider as `OPEN` (skip) for 60 seconds.
- After 60s: move to `HALF_OPEN` (try one request). Success → `CLOSED`. Failure → `OPEN` again.
- Log all state transitions with structlog.

In `orion/tools/mcp_client.py`:
- Wrap `call_tool` with a retry: up to 3 attempts with 1s exponential backoff.
- After 3 failures: return `Result.failure("MCP server unavailable")` and log.

## Step 3 — Rollback Integration
In `orion/safety/rollback.py`:
- `checkpoint(state: dict)` — push state snapshot to LIFO stack (max 50 entries).
- `rollback()` → pop latest, undo last OS/file action if reversible.
- Undo strategies:
  - File write → delete written file.
  - File delete → restore from snapshot bytes stored at checkpoint.
  - Git commit → `git revert HEAD`.
- Expose `POST /v1/tasks/{id}/rollback` endpoint.

## Step 4 — Structured Logging
In `orion/observability/logger.py`:
- All agents log with context: `log.info("event", agent="Planner", task_id=..., subtask_id=...)`.
- In production (`ORION_ENV=production`): output JSON via structlog's `JSONRenderer`.
- In dev: use `ConsoleRenderer` with color.

## Step 5 — Eval Suite Expansion
Add to `scripts/eval_task.py`, 25 tasks covering:
- OS: list files, read file, write file, execute shell command, find pattern in file
- Browser: open URL, click link, fill form, extract page text, take screenshot
- GitHub: list repos, create issue, open PR, clone repo, read file from repo
- Vision: describe screen, click element by description, type in focused field
- Multi-step: "Clone repo X, read README, create a summary file, commit it"
- Error cases: invalid path, permission denied, network failure (verify graceful failure)

## Step 6 — Prometheus Metrics
Rename `infra/prometheus/openclaw_rules.yaml` → `infra/prometheus/orion_rules.yaml`.
Update all metric names from `openclaw_*` to `orion_*`.
Add metrics:
- `orion_task_duration_seconds` (histogram)
- `orion_subtasks_parallel_count` (gauge)
- `orion_vision_api_latency_seconds` (histogram)
- `orion_llm_provider_failures_total` (counter, label: provider)

## Validation
- `make eval` — all 25 tasks pass.
- `make health` — all dependencies green.
- Stream a multi-step task and verify all SSE events arrive in order.
- Trigger a failure mid-task and verify rollback restores state.
```

---

## PHASE 6 — Packaging, CI/CD & v1.0 Release
**Goal:** Production container, GitHub Actions CI, K8s manifests, release tagging.

```markdown
# ORION v1.0 — Phase 6: Packaging & Release

## Objectives
1. Final Podman image build and push.
2. GitHub Actions CI pipeline.
3. K8s manifests updated for v1.0.
4. Version bump and changelog.

## Step 1 — Podman Image
Update `Dockerfile`:
```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    nodejs npm \
    scrot xdotool \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen --no-dev

COPY . .

ENV ORION_ENV=production
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8080/ready || exit 1
CMD ["uv", "run", "uvicorn", "orion.api.server:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Step 2 — GitHub Actions
Update `.github/workflows/ci.yml`:
```yaml
name: ORION CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - name: Install uv
        run: pip install uv
      - name: Install deps
        run: uv sync --frozen
      - name: Lint
        run: uv run ruff check orion/
      - name: Type check
        run: uv run mypy orion/ --ignore-missing-imports
      - name: Unit tests
        run: uv run pytest tests/unit/ -v
      - name: Integration tests
        run: uv run pytest tests/integration/ -v
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Podman image
        run: podman build -t orion:${{ github.sha }} .
      - name: Tag as latest
        run: podman tag orion:${{ github.sha }} ghcr.io/${{ github.repository_owner }}/orion:latest
```

## Step 3 — K8s Manifests
Update `infra/k8s/deployment.yaml`:
- Image: `ghcr.io/yourorg/orion:1.0.0`
- Add env var `VISION_API_URL` from Secret `orion-secrets`.
- Resource limits: `cpu: "2"`, `memory: "4Gi"`.

Update `infra/k8s/hpa.yaml`:
- `minReplicas: 2`, `maxReplicas: 10`
- Scale on `orion_task_duration_seconds` (custom metric via Prometheus adapter).

## Step 4 — Version & Changelog
- Update `pyproject.toml`: `version = "1.0.0"`.
- Create `CHANGELOG.md`:
  ```
  # ORION v1.0.0
  
  ## What's New
  - AgentScope-powered 4-agent HiClaw pipeline
  - Open-source MCP stdio tools (no Composio)
  - AsyncIO DAG parallel execution
  - Qwen2.5-VL vision via Colab + ngrok
  - ChromaDB embedded long-term memory
  - Podman-native container stack
  - 25-task eval suite
  - Full SSE streaming with circuit breakers
  - pyautogui computer control
  ```
- Tag release: `git tag v1.0.0 && git push origin v1.0.0`

## Final Validation Checklist
- [ ] `make test` — 176+ tests pass
- [ ] `make eval` — 25/25 eval tasks pass
- [ ] `make health` — all green
- [ ] `podman-compose up` — stack starts, /ready returns 200
- [ ] `uv run orion` — CLI starts, REPL works
- [ ] Vision task end-to-end (requires Colab running)
- [ ] SSE stream shows all events
- [ ] Rollback works on failed task
- [ ] Prometheus metrics visible in Grafana
- [ ] GitHub Actions CI green
```

---

*ORION v1.0 — Built with AgentScope · Open-source MCP · Qwen2.5-VL · Podman*
