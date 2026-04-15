# ORION v1.0 — Architecture Plan & Core System Design

---

## 1. System Overview

ORION v1.0 is a **locally-first, screen-aware, production-grade OS automation agent** built on four design principles:

1. **No cloud gatekeepers** — all tool execution flows through open-source MCP stdio servers, not cloud proxies
2. **Parallel-first execution** — subtasks are a DAG; independent nodes run with `asyncio.gather`
3. **Vision-native** — Qwen2.5-VL provides screen understanding; `pyautogui` provides actuation
4. **AgentScope-native** — all four HiClaw agents extend `agentscope.agents.AgentBase`

---

## 2. Component Architecture

### 2.1 HiClaw Pipeline (AgentScope)

```
User Instruction
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                  HiClaw Pipeline                        │
│                                                         │
│  Planner ──→ Executor (parallel DAG) ──→ Verifier       │
│     ▲                                       │           │
│     └────── Supervisor (retry loop) ◄───────┘           │
└─────────────────────────────────────────────────────────┘
```

Each agent is an `AgentBase` subclass with a role-specific Jinja2 system prompt.
The Supervisor decides between `complete`, `partial_complete`, and `retry` (re-routes to Planner with failure context).

### 2.2 Adaptive LLM Router

Priority chain with circuit breakers:
1. **vLLM** (local GPU — zero cost, maximum privacy)
2. **Groq** (fast cloud inference, rate-limited)
3. **OpenRouter** (fallback, broadest model coverage)

Each provider tracks consecutive failures. After 3 failures → OPEN state (60s skip). Circuit state is logged to Prometheus.

### 2.3 Open-Source MCP Tool Stack

| Category | MCP Server | Transport |
|---|---|---|
| os_tools | @modelcontextprotocol/server-filesystem | stdio |
| browser_tools | @playwright/mcp | stdio |
| github_tools | @modelcontextprotocol/server-github | stdio |
| vision_tools | Qwen2.5-VL (Colab + ngrok) | HTTP REST |

All MCP clients use the official `mcp` Python SDK (`mcp.client.stdio.stdio_client`).
The `ToolRegistry` lazy-initializes each client and caches the session.

### 2.4 Vision Layer

```
Screenshot (PIL.ImageGrab)
      │
      ▼
base64 encode
      │
      ▼  HTTP POST /analyze
Qwen2.5-VL (Colab GPU)
      │
      ▼  JSON: {result: "UI description + bounding boxes"}
Coordinate parser (regex)
      │
      ▼
pyautogui.click(x, y)
```

The Colab notebook runs Qwen2.5-VL-7B-Instruct exposed via a Flask server tunneled through ngrok.
ORION stores the ngrok URL in `VISION_API_URL` env var; the tool falls back with a clear error if unset.

### 2.5 Safety Shield

```
Tool call request
      │
      ├─ Permission gate (YAML manifest check) → DENY if blocked pattern
      │
      ├─ Risk classifier → LOW (auto-execute) | HIGH (HITL approval)
      │
      ├─ Rollback checkpoint (snapshot before destructive ops)
      │
      └─ Sandbox subprocess (memory + timeout caps)
```

### 2.6 Two-Tier Memory

| Tier | Implementation | Scope | Persistence |
|---|---|---|---|
| Working memory | Sliding window context | Current task session | In-process |
| Long-term memory | ChromaDB (embedded) | Cross-session | Disk at `.orion_memory/` |

ChromaDB replaces Qdrant — zero external service dependency. The embedder uses `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed).

### 2.7 Observability (TLP Stack)

- **Tracing**: structlog with `bind_contextvars(task_id=..., agent=...)` for every agent call
- **Logging**: JSON in production (`ORION_ENV=production`), colored console in dev
- **Prometheus**: 12 custom metrics covering latency, parallelism, vision, LLM failover
- **Grafana**: dashboard at `infra/grafana/dashboard.json`

---

## 3. File Structure (v1.0)

```
ORION/
├── .github/
│   └── workflows/
│       └── ci.yml                         # Podman build + pytest + ruff + mypy
│
├── scripts/
│   ├── colab_vision_server.ipynb          # NEW: Qwen2.5-VL + ngrok Colab notebook
│   ├── eval_task.py                       # 25-task eval suite (expanded from 10)
│   ├── healthcheck.py                     # Dependency health check
│   ├── seed_registry.py                   # Seed ChromaDB with example memories
│   └── start_vllm.sh                      # Local vLLM server launcher
│
├── configs/
│   ├── agents/
│   │   ├── executor.yaml                  # Max parallel subtasks, timeout
│   │   ├── planner.yaml                   # DAG output schema, max subtasks
│   │   ├── supervisor.yaml                # Retry policy, max retries
│   │   └── verifier.yaml                  # Validation strictness
│   ├── llm/
│   │   ├── router.yaml                    # Provider chain + circuit breaker config
│   │   ├── groq.yaml
│   │   ├── openrouter.yaml
│   │   └── vllm.yaml
│   ├── mcp/
│   │   ├── servers.yaml                   # NEW: open-source MCP server definitions
│   │   ├── browser.yaml                   # Playwright MCP config
│   │   ├── github.yaml                    # GitHub MCP config
│   │   └── os_automation.yaml             # Filesystem MCP config
│   ├── memory/
│   │   ├── longterm.yaml                  # ChromaDB persist_path, collection name
│   │   └── working.yaml                   # Sliding window size
│   └── safety/
│       ├── permissions.yaml               # Tool allowlists + HITL triggers
│       └── sandbox.yaml                   # Timeout, memory cap, gate mode
│
├── infra/
│   ├── grafana/
│   │   └── dashboard.json
│   ├── k8s/
│   │   ├── deployment.yaml                # Updated for v1.0 image
│   │   ├── hpa.yaml                       # Custom Prometheus metric scaling
│   │   └── service.yaml
│   └── prometheus/
│       └── orion_rules.yaml               # RENAMED from openclaw_rules.yaml
│
├── orion/
│   ├── __init__.py
│   │
│   ├── agentscope_config.py               # NEW: AgentScope init, model config mapping
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                        # UPDATED: now wraps AgentBase
│   │   ├── executor.py                    # UPDATED: DAG subtask executor, vision hook
│   │   ├── planner.py                     # UPDATED: outputs depends_on DAG JSON
│   │   ├── supervisor.py                  # UPDATED: AgentBase, retry logic
│   │   └── verifier.py                    # UPDATED: AgentBase
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── status.py                  # UPDATED: ChromaDB health, no Qdrant
│   │   │   ├── tasks.py                   # UPDATED: full SSE event taxonomy
│   │   │   └── tools.py
│   │   ├── schemas.py
│   │   └── server.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py
│   │   ├── result.py
│   │   └── task.py
│   │
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py             # NEW: CircuitBreaker class
│   │   ├── health.py
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── groq_provider.py
│   │   │   ├── openrouter_provider.py
│   │   │   └── vllm_provider.py
│   │   ├── quota.py
│   │   └── router.py                      # UPDATED: circuit breaker integration
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── embedder.py                    # sentence-transformers local embedder
│   │   ├── longterm.py                    # REPLACED: ChromaDB PersistentClient
│   │   ├── retriever.py
│   │   └── working.py
│   │
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── logger.py                      # structlog JSON/console dual mode
│   │   ├── metrics.py                     # UPDATED: orion_* metric names
│   │   └── tracer.py
│   │
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── dispatcher.py                  # REPLACED: asyncio DAG executor
│   │   ├── model_wrapper.py
│   │   ├── pipeline.py                    # UPDATED: AgentScope message routing
│   │   └── rpc_server.py
│   │
│   ├── safety/
│   │   ├── __init__.py
│   │   ├── gate.py
│   │   ├── manifest.py
│   │   ├── rollback.py                    # UPDATED: file/git undo strategies
│   │   └── sandbox.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── categories/
│       │   ├── __init__.py
│       │   ├── browser_tools.py           # MCP stdio via Playwright
│       │   ├── github_tools.py            # MCP stdio via server-github
│       │   ├── os_tools.py                # MCP stdio via server-filesystem
│       │   ├── saas_tools.py              # Local stub (extensible)
│       │   └── vision_tools.py            # NEW: screenshot + Qwen2.5VL + pyautogui
│       ├── mcp_client.py                  # REPLACED: MCPStdioClient (mcp SDK)
│       ├── registry.py                    # REPLACED: open-source server registry
│       └── selector.py
│
├── prompts/
│   ├── executor_system.j2                 # UPDATED: parallel task instructions
│   ├── planner_system.j2                  # UPDATED: DAG JSON output schema
│   ├── supervisor_system.j2
│   └── verifier_system.j2
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   │   ├── mock_tool_responses.json
│   │   └── sample_tasks.json
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_api.py
│   │   ├── test_mcp_stdio.py              # NEW: replaces test_composio_mcp.py
│   │   ├── test_memory.py                 # UPDATED: ChromaDB in-memory mock
│   │   └── test_pipeline.py
│   └── unit/
│       ├── __init__.py
│       ├── test_circuit_breaker.py        # NEW
│       ├── test_dag_dispatcher.py         # NEW
│       ├── test_exceptions.py
│       ├── test_executor.py               # UPDATED: AgentScope mocks
│       ├── test_longterm_memory.py        # UPDATED: ChromaDB mocks
│       ├── test_planner.py                # UPDATED: DAG output validation
│       ├── test_registry.py               # UPDATED: MCP stdio mocks
│       ├── test_result.py
│       ├── test_router.py
│       ├── test_safety.py
│       ├── test_supervisor.py
│       ├── test_task.py
│       ├── test_verifier.py
│       ├── test_vision_tools.py           # NEW
│       └── test_working_memory.py
│
├── orion_cli.py                           # NEW: rich + prompt_toolkit REPL
├── CHANGELOG.md                           # NEW
├── .dockerignore
├── .env.example                           # UPDATED: VISION_API_URL, GITHUB_PAT
├── .gitignore
├── Dockerfile                             # UPDATED: nodejs for MCP, scrot for screenshots
├── Makefile                               # UPDATED: podman-* targets
├── README.md                              # UPDATED: v1.0 docs
├── docker-compose.yml                     # DEPRECATED (kept for reference)
├── podman-compose.yml                     # NEW: replaces docker-compose
├── pyproject.toml                         # UPDATED: agentscope, mcp, chromadb, rich
└── uv.lock
```

---

## 4. Core Data Flows

### 4.1 Task Submission Flow
```
POST /v1/tasks {"instruction": "..."}
  → TaskManager creates Task(id, status=PENDING)
  → LLM Router selects provider (vLLM → Groq → OpenRouter)
  → PlannerAgent.reply(Msg(instruction)) → Msg(subtask_DAG_json)
  → dispatcher.execute_dag(subtasks, executor)
      → asyncio.gather(*[executor.execute_subtask(t) for t in ready_tier])
          → safety.gate.check(tool_call)
          → registry.call(category, tool_name, args)  ← MCP stdio or vision HTTP
          → rollback.checkpoint(state)
  → VerifierAgent.reply(Msg(results)) → Msg(pass/fail + reason)
  → SupervisorAgent.reply(Msg(verification)) → Msg(complete|retry|partial)
  → Task.status = COMPLETE | FAILED
  → SSE stream: event: done
```

### 4.2 Vision Computer Control Flow
```
Planner subtask: {tool_category: "vision_tools", tool_name: "click_element", arguments: {description: "Submit button"}}
  → executor.execute_subtask()
  → vision_tools.click_element("Submit button")
  → vision_tools.take_screenshot() → PNG bytes
  → base64 encode
  → HTTP POST VISION_API_URL/analyze (Colab Qwen2.5-VL)
  → parse JSON coordinates from response
  → pyautogui.click(x, y)
  → return {success: true, clicked_at: {x, y}}
```

### 4.3 Memory Retrieval Flow
```
PlannerAgent start:
  1. working_memory.get_context() → last N messages (in-process)
  2. longterm_memory.retrieve(instruction, n=5)
       → chromadb.Collection.query(query_texts=[instruction], n_results=5)
       → returns past successful subtask patterns
  3. Inject both into planner_system.j2 template context
  4. LLM generates subtask DAG informed by past successes
```

---

## 5. Key Design Decisions

### Why AgentScope over LangChain/LlamaIndex?
- Native multi-agent conversation protocol (Msg-passing)
- First-class async support
- Lightweight — no opinionated vector store or retriever abstractions
- Plays well with custom tool registries

### Why ChromaDB over Qdrant?
- Zero external service dependency — single `pip install chromadb` + one line to initialize
- Embedded mode: no Docker container, no port, no health check
- Sufficient for local/single-node ORION deployments; can be swapped for Qdrant in multi-node K8s

### Why stdio MCP over Composio?
- No API rate limits from third-party cloud
- Full offline operation
- Standard protocol — any new MCP server (community or custom) plugs in with 3 lines of config
- Cost: $0 per tool call

### Why Podman over Docker?
- Rootless by default (no daemon running as root)
- Drop-in `docker` CLI compatibility (`alias docker=podman` works)
- Better for K8s (generates K8s YAML from `podman generate kube`)
- No daemon process — lower memory footprint

### Why Qwen2.5-VL over proprietary vision APIs?
- Free on Colab T4/A100 — no per-image API cost
- 7B parameter model fits in 16GB VRAM with float16
- Strong OCR + UI element detection
- ngrok provides stable public URL for ORION to call

---

## 6. Configuration Reference

### `.env` (complete for v1.0)
```bash
# LLM Providers
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
VLLM_BASE_URL=http://localhost:8000  # Optional: local GPU server

# Tool Access
GITHUB_PAT=your_github_personal_access_token

# Vision
VISION_API_URL=https://xxxx.ngrok.io  # From Colab notebook output
PYAUTOGUI_FAILSAFE=true               # Move mouse to corner to abort

# Runtime
ORION_ENV=development                  # or: production
MAX_CONCURRENT_TASKS=5
MAX_PARALLEL_SUBTASKS=8
MCP_TIMEOUT_SECONDS=30

# Memory
CHROMA_PERSIST_PATH=.orion_memory
```

### `configs/llm/router.yaml`
```yaml
circuit_breaker:
  failure_threshold: 3
  open_duration_seconds: 60

providers:
  - name: vllm
    type: post_api_chat
    model: meta-llama/Llama-3.1-8B-Instruct
    base_url: "${VLLM_BASE_URL}/v1"
    priority: 1

  - name: groq
    type: openai_chat
    model: llama-3.1-70b-versatile
    api_key: "${GROQ_API_KEY}"
    base_url: https://api.groq.com/openai/v1
    priority: 2

  - name: openrouter
    type: openai_chat
    model: anthropic/claude-3-haiku
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api/v1
    priority: 3
```

---

## 7. AgentScope Integration Pattern

```python
# orion/agentscope_config.py
import agentscope

def init_orion_agentscope():
    agentscope.init(
        model_configs=[
            {
                "model_type": "post_api_chat",
                "config_name": "vllm_local",
                "api_url": "http://localhost:8000/v1/chat/completions",
                "headers": {},
            },
            {
                "model_type": "openai_chat",
                "config_name": "groq_fast",
                "model_name": "llama-3.1-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
                "client_args": {"base_url": "https://api.groq.com/openai/v1"},
            },
        ],
        project="ORION",
        logger_level="INFO",
    )


# orion/agents/planner.py
from agentscope.agents import AgentBase
from agentscope.message import Msg

class PlannerAgent(AgentBase):
    def __init__(self, working_mem, longterm_mem):
        super().__init__(
            name="Planner",
            sys_prompt=self._build_system_prompt(),
            model_config_name="groq_fast",
        )
        self.working_mem = working_mem
        self.longterm_mem = longterm_mem

    def _build_system_prompt(self) -> str:
        # Load and render prompts/planner_system.j2
        ...

    def reply(self, x: Msg) -> Msg:
        context = self.working_mem.get_context()
        memories = self.longterm_mem.retrieve(x.content)
        # Augment message with context
        augmented = Msg(
            name="user",
            content=f"Context: {context}\nMemories: {memories}\nTask: {x.content}"
        )
        response = self.model(self.format(augmented))
        subtask_dag = json.loads(response.text)
        return Msg(name="Planner", content=subtask_dag, role="assistant")
```

---

## 8. Eval Task Examples (25-task suite)

| # | Category | Instruction | Expected |
|---|---|---|---|
| 1 | OS | "List all Python files in /workspace recursively" | File list |
| 2 | OS | "Read the file README.md and return its content" | File content |
| 3 | OS | "Write 'Hello ORION' to /tmp/test.txt" | File created |
| 4 | OS | "Find all lines containing 'TODO' in /workspace" | Line list |
| 5 | OS | "Count files by extension in /workspace" | Dict |
| 6 | Browser | "Open https://example.com and return the page title" | Title string |
| 7 | Browser | "Extract all links from https://news.ycombinator.com" | URL list |
| 8 | GitHub | "List my repositories" | Repo list |
| 9 | GitHub | "Create issue titled 'Test from ORION' in repo X" | Issue URL |
| 10 | GitHub | "Read the contents of README.md from repo X" | File content |
| 11 | Vision | "Take a screenshot and describe the screen" | Description |
| 12 | Vision | "Click the button labeled 'OK' on screen" | Click success |
| 13 | Vision | "Type 'Hello World' in the focused text field" | Type success |
| 14 | Multi-step | "List files, find largest, copy it to /tmp" | Copy success |
| 15 | Multi-step | "Read config.yaml, extract the port value, write it to /tmp/port.txt" | File match |
| 16-25 | Error cases | Invalid paths, no permissions, network failures | Graceful failure + rollback |

---

*ORION v1.0 Architecture · AgentScope · MCP · Qwen2.5-VL · ChromaDB · Podman*
