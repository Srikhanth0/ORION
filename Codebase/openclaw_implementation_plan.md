# OpenClaw — Step-by-Step Implementation Plan

> **Goal**: Build a production-grade OS automation agent using AgentScope, HiClaw,
> Composio MCP (40+ tools), and a tiered LLM backend (Qwen 2.5/vLLM → Groq → OpenRouter).

---

## Phase 0 — Environment & Prerequisites (Day 1–2)

### Step 0.1 — System requirements check

```bash
# Python 3.11+, CUDA 12+ (for vLLM), 24 GB VRAM min for Qwen-72B-Q4
python --version && nvidia-smi
# Node 18+ for Composio MCP server
node --version
```

### Step 0.2 — Clone and scaffold

```bash
mkdir openclaw && cd openclaw
git init
uv init --name openclaw  # or: poetry init
uv add agentscope composio-core composio-mcp fastapi uvicorn \
        qdrant-client sentence-transformers langsmith \
        prometheus-client structlog httpx pydantic
uv add --dev pytest pytest-asyncio pytest-mock ruff mypy
```

### Step 0.3 — Create `.env`

```env
# LLM providers
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=deepseek/deepseek-chat

# Composio
COMPOSIO_API_KEY=...

# Memory
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=openclaw_tasks

# Observability
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=openclaw
```

---

## Phase 1 — LLM Router (Day 3–5)

### Step 1.1 — Write individual provider wrappers

Each provider wraps the same `chat()` interface so the router is provider-agnostic.

```python
# openclaw/llm/providers/vllm_provider.py
from openai import AsyncOpenAI

class VLLMProvider:
    def __init__(self, cfg: dict):
        self.client = AsyncOpenAI(
            base_url=cfg["base_url"], api_key="not-needed"
        )
        self.model = cfg["model"]

    async def chat(self, messages: list[dict], **kwargs) -> str:
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, **kwargs
        )
        return resp.choices[0].message.content

    async def is_healthy(self) -> bool:
        try:
            await self.client.models.list()
            return True
        except Exception:
            return False
```

Repeat the same pattern for `GroqProvider` (uses `groq` SDK) and `OpenRouterProvider` (uses `httpx` to the OpenRouter endpoint). Add `remaining_quota()` to `GroqProvider` by parsing the `x-ratelimit-remaining-requests` response header.

### Step 1.2 — Adaptive router with fallback chain

```python
# openclaw/llm/router.py
class AdaptiveLLMRouter:
    def __init__(self, providers: list):
        self.chain = providers  # [vllm, groq, openrouter]

    async def chat(self, messages, **kwargs) -> str:
        for provider in self.chain:
            if await provider.is_healthy():
                try:
                    return await provider.chat(messages, **kwargs)
                except Exception as e:
                    logger.warning(f"{provider.__class__.__name__} failed: {e}")
        raise RuntimeError("All LLM providers exhausted")
```

### Step 1.3 — Test router fallback

```bash
# Kill vLLM, exhaust Groq quota → confirm OpenRouter kicks in
pytest tests/unit/test_router.py -v
```

---

## Phase 2 — AgentScope Setup (Day 6–8)

### Step 2.1 — Install and configure AgentScope

```python
# openclaw/orchestrator/pipeline.py
import agentscope
from agentscope.pipelines import SequentialPipeline

def init_agentscope(router: AdaptiveLLMRouter):
    agentscope.init(
        model_configs=[{
            "config_name": "openclaw-llm",
            "model_type": "openai_chat",
            "model_name": "openclaw-router",
            # Hook our router into AgentScope's model calling
        }],
        use_monitor=True,
        logger_level="INFO",
    )
```

AgentScope's `model_configs` accepts custom model wrappers — subclass `agentscope.models.ModelWrapperBase` and delegate `__call__` to `router.chat()`.

### Step 2.2 — Define the agent pipeline topology

```python
# Two pipeline modes:
# 1. Simple task: User → Planner → Executor → Verifier → Done
# 2. Complex task: User → Planner → [Executor × N subtasks in parallel] → Verifier → Done

pipeline = SequentialPipeline([planner, executor, verifier])
# For parallel subtasks:
from agentscope.pipelines import ForLoopPipeline
```

### Step 2.3 — Message bus schema

Every inter-agent message extends AgentScope's `Msg` with an `openclaw_meta` field carrying `task_id`, `subtask_id`, `step_index`, and `rollback_point`.

---

## Phase 3 — HiClaw Agents (Day 9–15)

### Step 3.1 — Base agent

```python
# openclaw/agents/base.py
from agentscope.agents import AgentBase
from openclaw.memory import WorkingMemory, LongTermMemory
from openclaw.observability import tracer

class BaseOpenClawAgent(AgentBase):
    def __init__(self, name, cfg, router, memory_cfg):
        super().__init__(name=name, model_config_name="openclaw-llm")
        self.router = router
        self.working_mem = WorkingMemory(cfg=memory_cfg["working"])
        self.long_mem = LongTermMemory(cfg=memory_cfg["longterm"])

    def reply(self, x):  # AgentScope entry point
        raise NotImplementedError
```

### Step 3.2 — Planner agent (ReAct + CoT → TaskDAG)

The Planner receives the raw user task and returns a `TaskDAG` — a list of `Subtask` nodes with dependency edges.

```python
# openclaw/agents/planner.py
SYSTEM_PROMPT = open("prompts/planner_system.j2").read()

class PlannerAgent(BaseOpenClawAgent):
    def reply(self, x: Msg) -> Msg:
        # 1. Retrieve similar past tasks from long-term memory
        similar = self.long_mem.retrieve(x.content, top_k=3)
        # 2. Build prompt: system + few-shot from memory + current task
        messages = self._build_messages(x.content, similar)
        # 3. Call LLM → structured JSON DAG
        raw = self.router.chat(messages, response_format={"type": "json_object"})
        dag = TaskDAG.model_validate_json(raw)
        return Msg(name=self.name, content=dag, role="assistant")
```

**`prompts/planner_system.j2`** instructs the LLM to output JSON like:
```json
{
  "subtasks": [
    {"id": "s1", "action": "clone repo", "tool": "github_clone", "depends_on": []},
    {"id": "s2", "action": "run tests",  "tool": "exec_cmd",     "depends_on": ["s1"]}
  ]
}
```

### Step 3.3 — Executor agent (tool-use loop)

```python
class ExecutorAgent(BaseOpenClawAgent):
    MAX_RETRIES = 3

    def reply(self, x: Msg) -> Msg:
        dag: TaskDAG = x.content
        results = []
        for subtask in dag.topological_order():
            for attempt in range(self.MAX_RETRIES):
                try:
                    tool = self.tool_registry.get(subtask.tool)
                    result = tool.invoke(subtask.params)
                    self.rollback_engine.checkpoint(subtask.id, result)
                    results.append(StepResult(subtask_id=subtask.id, ok=True, output=result))
                    break
                except ToolError as e:
                    if attempt == self.MAX_RETRIES - 1:
                        results.append(StepResult(subtask_id=subtask.id, ok=False, error=str(e)))
        return Msg(name=self.name, content=results, role="assistant")
```

### Step 3.4 — Verifier agent

Verifier receives `List[StepResult]` and runs assertions defined in the `TaskDAG` plus an LLM self-critique pass. Returns `VerificationReport` with per-step pass/fail.

### Step 3.5 — Supervisor agent

Supervisor receives a `VerificationReport` with failures. It decides: **retry subtask** (if transient error) | **rollback** (if state corrupted) | **ask human** (if ambiguous) | **abort** (if safety violation). HITL is implemented as a simple blocking `input()` in dev mode and a webhook POST in production.

---

## Phase 4 — Composio MCP Integration (Day 16–20)

### Step 4.1 — Authenticate Composio

```bash
pip install composio-core composio-mcp
composio login          # opens browser OAuth
composio apps enable github
composio apps enable shell
composio apps enable browserbase
composio apps enable slack notion linear gmail
```

### Step 4.2 — Build the ToolRegistry

```python
# openclaw/tools/registry.py
from composio import ComposioToolSet

class ToolRegistry:
    def __init__(self, cfg: dict):
        self.toolset = ComposioToolSet(api_key=cfg["api_key"])
        self._tools: dict[str, ComposioTool] = {}

    def load(self, enabled_apps: list[str]):
        for app in enabled_apps:
            tools = self.toolset.get_tools(apps=[app])
            for t in tools:
                self._tools[t.name] = t

    def get(self, name: str) -> ComposioTool:
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' not in registry")
        return self._tools[name]

    def score(self, subtask_description: str) -> list[tuple[str, float]]:
        # Simple keyword scoring; replace with embedding similarity
        scores = []
        for name, tool in self._tools.items():
            score = len(set(subtask_description.lower().split()) &
                        set(tool.description.lower().split()))
            scores.append((name, score))
        return sorted(scores, key=lambda x: -x[1])
```

### Step 4.3 — MCP tool invocation with safety gate

```python
# openclaw/tools/mcp_client.py
class MCPClient:
    def __init__(self, registry, safety_manifest):
        self.registry = registry
        self.manifest = safety_manifest

    def invoke(self, tool_name: str, params: dict) -> dict:
        # 1. Check permission manifest
        self.manifest.check(tool_name, params)  # raises SafetyError if denied
        # 2. Check if destructive → require supervisor gate
        if self.manifest.is_destructive(tool_name):
            self.gate.approve(tool_name, params)
        # 3. Execute
        tool = self.registry.get(tool_name)
        return tool.execute(params)
```

### Step 4.4 — Register tools in AgentScope

AgentScope supports passing tools as JSON-schema `service_functions`. Wrap each `ComposioTool` to emit the correct schema and wire them into the Executor's model call.

---

## Phase 5 — Memory Layer (Day 21–24)

### Step 5.1 — Working memory (in-context)

AgentScope's built-in `memory` module handles the sliding-window context. Configure max tokens in `configs/memory/working.yaml`. Add a `summarize_on_overflow` hook that calls the LLM router to compress old context before evicting.

### Step 5.2 — Long-term memory with Qdrant

```python
# openclaw/memory/longterm.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    def __init__(self, cfg):
        self.client = QdrantClient(url=cfg["url"])
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = cfg["collection"]

    def store(self, task_description: str, execution_plan: dict):
        vec = self.encoder.encode(task_description).tolist()
        self.client.upsert(self.collection, points=[{
            "id": str(uuid4()),
            "vector": vec,
            "payload": {"task": task_description, "plan": execution_plan}
        }])

    def retrieve(self, query: str, top_k=3) -> list[dict]:
        vec = self.encoder.encode(query).tolist()
        hits = self.client.search(self.collection, vec, limit=top_k)
        return [h.payload for h in hits]
```

After every successful task run, the Supervisor stores the final `TaskDAG` + `List[StepResult]` in long-term memory so the Planner can reuse proven plans.

---

## Phase 6 — Safety Layer (Day 25–27)

### Step 6.1 — Permission manifest

`configs/safety/permissions.yaml` maps tool categories to allowed operations:

```yaml
github:
  allowed: [create_issue, list_prs, get_file_content, push_files]
  denied:  [delete_repo, transfer_repo]
shell:
  allowed: [exec_cmd]
  denied_patterns: ["rm -rf /", "sudo rm", "dd if="]
  max_timeout_seconds: 30
filesystem:
  allowed_paths: ["/home", "/tmp", "/workspace"]
  denied_paths:  ["/etc", "/sys", "/boot"]
```

### Step 6.2 — Rollback engine

Before each Executor step, create a lightweight checkpoint:
- **File operations**: store file hash + content snapshot in `/tmp/openclaw_checkpoints/`
- **Shell commands**: record working directory state with `git stash` if inside a repo
- **API calls**: mark as non-reversible; Supervisor must confirm before execution

On rollback, replay checkpoints in reverse order.

---

## Phase 7 — API + Observability (Day 28–30)

### Step 7.1 — FastAPI server

```python
# openclaw/api/server.py
from fastapi import FastAPI
from openclaw.orchestrator import OpenClawPipeline

app = FastAPI(title="OpenClaw API")

@app.post("/tasks")
async def create_task(req: TaskRequest) -> TaskResponse:
    pipeline = OpenClawPipeline.from_config("configs/")
    result = await pipeline.run(req.instruction)
    return TaskResponse(task_id=result.task_id, status=result.status)

@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> TaskStatusResponse:
    ...
```

### Step 7.2 — Prometheus metrics

```python
# openclaw/observability/metrics.py
from prometheus_client import Counter, Histogram

llm_requests   = Counter("openclaw_llm_requests_total", "LLM calls", ["provider"])
llm_latency    = Histogram("openclaw_llm_latency_seconds", "LLM latency", ["provider"])
tool_calls     = Counter("openclaw_tool_calls_total", "Tool calls", ["tool_name"])
task_cost_usd  = Counter("openclaw_task_cost_usd_total", "Estimated cost")
```

### Step 7.3 — LangSmith tracing

```python
from langsmith import traceable

@traceable(name="planner-agent")
def planner_reply(messages):
    ...
```

---

## Phase 8 — Docker + vLLM Setup (Day 31–33)

### Step 8.1 — Start Qwen 2.5 via vLLM

```bash
# scripts/start_vllm.sh
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4 \
  --max-model-len 32768 \
  --tensor-parallel-size 2 \
  --quantization gptq \
  --served-model-name qwen2.5-72b
```

For machines without 24 GB VRAM, use `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4` as a drop-in.

### Step 8.2 — Docker Compose full stack

```yaml
# docker-compose.yml
services:
  openclaw-api:
    build: .
    ports: ["8080:8080"]
    env_file: .env
    depends_on: [qdrant, prometheus]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]

  prometheus:
    image: prom/prometheus:latest
    volumes: ["./infra/prometheus:/etc/prometheus"]
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    volumes: ["./infra/grafana:/etc/grafana/provisioning"]

volumes:
  qdrant_data:
```

---

## Phase 9 — Testing & Hardening (Day 34–38)

### Step 9.1 — Unit tests (mocked LLM + tools)

```bash
pytest tests/unit/ -v --tb=short
```

Mock the LLM router with `pytest-mock` to return pre-baked JSON plans. Mock Composio MCP tools with fixture payloads in `tests/fixtures/mock_tool_responses.json`.

### Step 9.2 — Integration smoke test

```bash
python scripts/healthcheck.py
# Runs: "create a file called hello.txt in /tmp with content 'hello world'"
# Expected: file exists, verifier passes, no safety violations
```

### Step 9.3 — Eval suite — 10 reference tasks

Define 10 canonical OS automation tasks in `tests/fixtures/sample_tasks.json` ranging from simple (copy a file) to complex (scaffold a new Python project from a spec). Run with `scripts/eval_task.py` and track pass rate + cost per task across LLM providers.

---

## Phase 10 — Production Hardening (Day 39–42)

| Concern | Solution |
|---|---|
| vLLM OOM | Set `--max-model-len` + `--gpu-memory-utilization 0.85`; auto-restart on OOM |
| Groq rate limits | Exponential backoff + quota cache in Redis; switch to OpenRouter at 80% quota |
| Runaway tasks | Global 5-minute task timeout; Supervisor abort + rollback on expiry |
| Concurrent users | AgentScope RPC server handles multiple pipelines; Qdrant handles concurrent reads |
| Secret leakage | Never log tool params in full; mask API keys in structured logs |
| Agent loops | Max 20 Executor iterations per task; Supervisor escalates on limit hit |

---

## Milestone summary

| Phase | Output | Days |
|---|---|---|
| 0 — Environment | Deps installed, `.env` set, scaffold created | 1–2 |
| 1 — LLM Router | 3-provider fallback chain with health checks | 3–5 |
| 2 — AgentScope | Pipeline + message bus wired | 6–8 |
| 3 — HiClaw agents | Planner, Executor, Verifier, Supervisor working | 9–15 |
| 4 — Composio MCP | 40+ tools registered, invocable from Executor | 16–20 |
| 5 — Memory | Working + long-term memory operational | 21–24 |
| 6 — Safety | Permission manifest, gate, rollback engine | 25–27 |
| 7 — API + Obs | FastAPI endpoint, Prometheus, LangSmith | 28–30 |
| 8 — Docker/vLLM | Full stack in Docker Compose | 31–33 |
| 9 — Testing | Unit + integration + eval suite | 34–38 |
| 10 — Hardening | Rate limits, timeouts, secret masking | 39–42 |

**Total: ~6 weeks to a production-ready v1.0.**


# ╔══════════════════════════════════════════════════════════════════╗
# ║  PROMPT 2.5 — MULTIMODAL ROUTER & RATE-LIMIT ENGINEER           ║
# ╚══════════════════════════════════════════════════════════════════╝

SYSTEM PROMPT — OPENCLAW MULTIMODAL ROUTER ENGINEER
===================================================

You are a specialist LLM infrastructure engineer working on OpenClaw's
adaptive multimodal routing layer. Your responsibility is to build a
highly resilient, vision-capable backend that parses OS UI screenshots 
and handles complex reasoning without hitting free-tier API rate limits.

Because the host machine is highly memory-constrained (8GB RAM), NO 
local vision models will be used. You must rely entirely on cloud APIs.

═══════════════════════════════════════════════════════════════
ACTIVE SKILLS & MODEL ARCHITECTURE
═══════════════════════════════════════════════════════════════

SKILL: Free-Tier Multimodal & Reasoning Assignment
  — You must implement exact routing to these specific models:
    1. FAST VISION (UI Screenshots): `meta-llama/llama-4-scout-17b-16e-instruct` via Groq. 
    2. HIGH REASONING (Planner): `openai/gpt-oss-120b:free` via OpenRouter.
    3. VISION FALLBACK: `google/gemma-4-26b-a4b-it:free` via OpenRouter.
    4. FAST TEXT (Executor): `llama-3.3-70b-versatile` via Groq.

SKILL: Dynamic Rate Limit & Header Tracking
  — Groq Free Tier enforces a strict 30 Requests Per Minute (RPM).
  — OpenRouter Free Tier enforces a strict 20 Requests Per Minute (RPM).
  — DO NOT rely solely on `time.sleep()`. 
  — In the GroqProvider, you must intercept the response headers and parse `x-ratelimit-remaining-requests` and `x-ratelimit-remaining-tokens`.
  — Cache these values in a `QuotaTracker` singleton. If `remaining-tokens` drops below 1,000, trigger an immediate, graceful fallback to OpenRouter before a 429 error occurs.

SKILL: Defensive Base Delays & Exponential Backoff
  — As a baseline safety measure, enforce a global asyncio delay queue:
      • Minimum 2.5s between Groq requests.
      • Minimum 3.5s between OpenRouter requests.
  — If a `429 Too Many Requests` error is thrown:
      1. Parse the `retry-after` header from the exception to wait the exact required seconds.
      2. If no header is present, trigger an exponential backoff: 2s → 4s → 8s.
      3. Trip the circuit breaker and switch providers after 3 failures.

SKILL: UI Screenshot Injection (Multimodal)
  — The `AgentScope` Msg objects will occasionally contain file paths to OS screenshots.
  — Your provider wrappers MUST detect image paths and convert them to base64.
  — Format the payload precisely as required by Groq and OpenRouter: 
    `{"type": "image_url", "image_url": "data:image/png;base64,<B64_STRING>"}`.
  — If a text-only model (like gpt-oss-120b) receives an image, the Router must automatically intercept it and route it to the vision model instead.

═══════════════════════════════════════════════════════════════
OUTPUT CONTRACT
═══════════════════════════════════════════════════════════════

Produce in this order:
  1. openclaw/llm/providers/base.py       — LLMProvider protocol + QuotaInfo
  2. openclaw/llm/providers/groq_provider.py — MUST include header tracking + vision parsing
  3. openclaw/llm/providers/openrouter_provider.py
  4. openclaw/llm/quota.py                — The QuotaTracker singleton
  5. openclaw/llm/router.py               — AdaptiveLLMRouter + CircuitBreaker + Vision Intercept
  6. tests/unit/test_rate_limits.py       — Mock 429s, test backoff logic, test header parsing.

Do not skip any file. Every module must be fully typed and explicitly handle 
the header parsing logic. NEVER generate placeholder code.