<p align="center">
  <img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+">
  <img src="https://img.shields.io/badge/framework-AgentScope-purple.svg" alt="AgentScope">
  <img src="https://img.shields.io/badge/tools-Composio_MCP-green.svg" alt="Composio MCP">
  <img src="https://img.shields.io/badge/api-FastAPI-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/vectors-Qdrant-red.svg" alt="Qdrant">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen.svg" alt="MIT">
</p>

# ORION — Autonomous OS Automation Agent

> **A hierarchical multi-agent system that decomposes natural language instructions into executable subtask DAGs, runs them through a defensive tool pipeline with permission gating and rollback, and verifies results before reporting.**

---

## Table of Contents

- [What Is ORION?](#what-is-orion)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [LLM Providers](#llm-providers)
  - [Safety & Permissions](#safety--permissions)
  - [Memory System](#memory-system)
  - [Agent Tuning](#agent-tuning)
  - [MCP Tool Registry](#mcp-tool-registry)
- [Usage](#usage)
  - [Starting the API Server](#starting-the-api-server)
  - [Submitting a Task](#submitting-a-task)
  - [Streaming Task Events](#streaming-task-events)
  - [Checking Task Status](#checking-task-status)
  - [Browsing Tools](#browsing-tools)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Production Deployment](#kubernetes-production-deployment)
- [Observability](#observability)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Makefile Reference](#makefile-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What Is ORION?

ORION is an autonomous agent that takes a plain-English instruction — *"Clone the repo, run the tests, and file a GitHub issue with the results"* — and executes it end-to-end on your OS. It:

1. **Plans** — Decomposes the instruction into a dependency-aware subtask DAG.
2. **Executes** — Runs each subtask using real tools (shell, GitHub, browser, Slack, Notion, etc.) via Composio MCP.
3. **Verifies** — Compares actual output against expected output for each step.
4. **Supervises** — Decides to mark the task as complete, retry failed steps, or escalate to a human.

Every tool call passes through a **safety layer** with permission manifests, destructive operation gates, sandboxed execution, and LIFO rollback — so the agent can never silently `rm -rf /` your system.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                          │
│         POST /v1/tasks  ·  GET /stream  ·  GET /tools            │
└─────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────▼───────────────┐
              │     Adaptive LLM Router       │
              │  vLLM → Groq → OpenRouter     │
              │  Circuit breakers · Failover  │
              └───────────────┬───────────────┘
                              │
         ┌────────────────────▼────────────────────┐
         │         HiClaw Agent Pipeline           │
         │                                         │
         │  ┌──────────┐    ┌──────────┐           │
         │  │ Planner  │───▶│ Executor │           │
         │  └──────────┘    └────┬─────┘           │
         │                       │                 │
         │  ┌──────────┐    ┌───▼──────┐           │
         │  │Supervisor│◀───│ Verifier │           │
         │  └──────────┘    └──────────┘           │
         └─────────┬───────────────────────────────┘
                   │
    ┌──────────────▼──────────────┐
    │      Safety Layer           │
    │  Permissions · Gate ·       │
    │  Sandbox · Rollback         │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │   Composio MCP Tools        │
    │  GitHub · Shell · Browser   │
    │  Slack · Notion · Gmail     │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │     Memory System           │
    │  Working (in-context)       │
    │  Long-term (Qdrant vectors) │
    └─────────────────────────────┘
```

---

## Features

| Category | What You Get |
|----------|-------------|
| **Multi-Provider LLM** | vLLM (local, zero-cost) → Groq (free tier) → OpenRouter (fallback). Automatic failover with circuit breakers. Role-based model assignment (fast_text, reasoning, vision). |
| **4-Agent Pipeline** | Planner (DAG decomposition) → Executor (tool execution) → Verifier (output validation) → Supervisor (retry/escalate/complete). Full `orion_meta` context propagation. |
| **28+ MCP Tools** | GitHub (7), OS/Shell (8), Browser (7), SaaS (6) — all with typed wrappers, schema validation, and category-based routing. |
| **Safety Layer** | YAML-driven permission manifests, shell command pattern blocking, filesystem path traversal prevention, destructive operation gating (auto/strict), sandboxed subprocess execution, LIFO rollback checkpoints. |
| **Two-Tier Memory** | Working memory with 80% token-budget eviction + LLM summarisation. Long-term memory via Qdrant with `all-MiniLM-L6-v2` semantic embeddings and cosine retrieval. |
| **Observability** | LangSmith distributed tracing, 12 Prometheus metric types, Grafana dashboards (4 rows), structured JSON logging with secret redaction. |
| **Production API** | FastAPI with SSE streaming, `asyncio.Semaphore` concurrency limits, RFC 7807 error responses, CORS, graceful shutdown. |
| **Deployment** | Multi-stage Dockerfile (<600MB), Docker Compose (4 services), Kubernetes manifests (Deployment, Service, HPA), GitHub Actions CI. |

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.13+ | Required |
| **uv** | Latest | [Install uv](https://docs.astral.sh/uv/) — replaces pip/venv |
| **Docker** | 24+ | For `docker compose` deployment |
| **Docker Compose** | v2+ | Included with modern Docker |
| **GPU** | Optional | Only needed if running vLLM locally |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-org/orion.git
cd orion

# 2. Install all dependencies
make install
# or: uv sync --all-extras

# 3. Create your environment file
cp .env.example .env

# 4. Edit .env with your API keys (see Configuration below)
```

---

## Configuration

ORION is configured through **environment variables** (`.env`) for secrets and **YAML files** (`configs/`) for everything else. You never need to edit Python source code to tune behavior.

### Environment Variables

Create `.env` from the template and fill in your keys:

```bash
cp .env.example .env
```

#### LLM Provider Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `VLLM_BASE_URL` | No | vLLM server URL (default: `http://localhost:8000/v1`). Only needed if running a local vLLM instance. |
| `VLLM_MODEL` | No | HuggingFace model ID for vLLM (default: `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4`) |
| `GROQ_API_KEY` | **Yes** | Free Groq API key from [console.groq.com](https://console.groq.com). Starts with `gsk_`. |
| `GROQ_TEXT_MODEL` | No | Groq text model (default: `llama-3.3-70b-versatile`) |
| `GROQ_VISION_MODEL` | No | Groq vision model (default: `meta-llama/llama-4-scout-17b-16e-instruct`) |
| `OPENROUTER_API_KEY` | Recommended | OpenRouter key from [openrouter.ai](https://openrouter.ai). Starts with `sk-or-`. Acts as pay-per-token fallback. |
| `OPENROUTER_REASONING_MODEL` | No | High-reasoning model (default: `openai/gpt-oss-120b:free`) |
| `OPENROUTER_VISION_MODEL` | No | Vision fallback (default: `google/gemma-4-26b-a4b-it:free`) |

> **Minimum viable setup:** You only need `GROQ_API_KEY` to get started. Everything else has sane defaults or graceful fallbacks.

#### Tool & Memory Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `COMPOSIO_API_KEY` | For tools | Composio key for MCP tool execution (GitHub, Slack, etc.) |
| `QDRANT_URL` | No | Vector DB URL (default: `http://localhost:6333`). Auto-connected when using Docker Compose. |
| `QDRANT_COLLECTION` | No | Collection name (default: `orion_tasks`) |

#### Observability Keys

| Variable | Required | Description |
|----------|----------|-------------|
| `LANGSMITH_API_KEY` | No | Enables LangSmith distributed tracing. Without it, traces go to structlog (dev-friendly). |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: `orion`) |

#### Runtime Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ORION_SAFETY_MODE` | `strict` | `strict` = all destructive ops need human approval. `permissive` = low-risk auto-approved. |
| `ORION_TASK_TIMEOUT_SECONDS` | `300` | Global per-task timeout (5 minutes). |
| `ORION_MAX_EXECUTOR_ITERATIONS` | `20` | Max steps the executor can take before being killed. |
| `MAX_CONCURRENT_TASKS` | `5` | Max tasks running simultaneously. Extra requests get `429 Retry-After`. |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins, comma-separated. |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. |
| `ORION_ENV` | `development` | Set to `production` for JSON log output (for Docker log drivers). |
| `METRICS_PORT` | `9091` | Port for the Prometheus `/metrics` endpoint. |

---

### LLM Providers

**File:** `configs/llm/router.yaml`

Controls the adaptive LLM routing — which providers to try and in what order:

```yaml
# Provider priority (first healthy provider wins)
fallback_chain:
  - vllm           # Priority 1: local GPU, zero cost
  - groq           # Priority 2: free tier, rate-limited
  - openrouter     # Priority 3: pay-per-token

# Role-based model assignment
roles:
  fast_text:                      # Used for: summarisation, simple completions
    preferred_provider: groq
    model: llama-3.3-70b-versatile
  reasoning:                      # Used for: planning, complex decisions
    preferred_provider: openrouter
    model: openai/gpt-oss-120b:free
  vision:                         # Used for: screenshot analysis, UI tasks
    preferred_provider: groq
    model: meta-llama/llama-4-scout-17b-16e-instruct

# Circuit breaker (auto-disables failing providers)
circuit_breaker:
  failure_threshold: 3            # 3 consecutive failures → trip
  recovery_timeout_seconds: 60    # Retry after 60s
```

Individual provider configs live in `configs/llm/vllm.yaml`, `groq.yaml`, and `openrouter.yaml`.

---

### Safety & Permissions

#### Permission Manifest

**File:** `configs/safety/permissions.yaml`

Defines what each tool category is **allowed** and **denied** from doing:

```yaml
github:
  allowed: [create_issue, list_prs, get_file_content, push_files, create_branch]
  denied:  [delete_repo, transfer_repo, update_org_settings]

shell:
  denied_patterns:
    - "rm -rf /"           # Destructive filesystem wipe
    - "sudo rm"            # Privileged deletion
    - "dd if="             # Raw disk write
    - "mkfs"               # Filesystem format
    - ":(){:|:&};:"        # Fork bomb

filesystem:
  allowed_paths: ["/home", "/tmp", "/workspace", "C:\\Users"]
  denied_paths:  ["/etc", "/sys", "/boot", "C:\\Windows\\System32"]
```

**To add a new allowed tool action**: Add it to the appropriate category's `allowed` list.
**To block a dangerous command pattern**: Add it to `shell.denied_patterns`.
**To restrict filesystem access**: Modify `allowed_paths` and `denied_paths`.

#### Sandbox & Gate

**File:** `configs/safety/sandbox.yaml`

Controls execution limits and the destructive operation gate:

```yaml
# Timeouts
task_timeout_seconds: 300       # 5 min per task
subtask_timeout_seconds: 30     # 30s per subtask

# Resource caps per subprocess
resource_limits:
  max_memory_mb: 512
  max_cpu_percent: 80

# Gate mode: "auto" or "strict"
#   auto   — low risk auto-approved, high risk needs approval
#   strict — ALL destructive ops require human approval
gate_mode: auto

# What counts as destructive
destructive_operations:
  - delete_file
  - delete_repo
  - exec_cmd
  - push_files
  - send_email
  - bulk_delete
```

---

### Memory System

#### Working Memory (In-Context)

**File:** `configs/memory/working.yaml`

Per-task ephemeral memory that feeds context into agent prompts:

```yaml
max_context_tokens: 8192     # Token budget per task
eviction_threshold: 0.80     # At 80% full → evict + summarise oldest entries
summarize_on_overflow: true  # Use LLM to summarise before evicting
summary_max_tokens: 512      # Max tokens for summary
summary_model_role: fast_text # Which LLM role for summarisation
```

#### Long-Term Memory (Qdrant)

**File:** `configs/memory/longterm.yaml`

Cross-task persistent memory using vector similarity search:

```yaml
backend: qdrant
url: http://localhost:6333
collection: orion_tasks           # Successful tasks
failure_collection: orion_failures # Failed tasks (never used as few-shot)

embedding_model: all-MiniLM-L6-v2 # 384-dim, fast, free
default_top_k: 3                   # Return top 3 similar past tasks
score_threshold: 0.5               # Minimum cosine similarity
```

Past successful tasks are automatically stored and retrieved as few-shot examples for future similar tasks.

---

### Agent Tuning

**Directory:** `configs/agents/`

Each agent has a YAML config controlling its behavior:

| File | Key Settings |
|------|-------------|
| `planner.yaml` | `max_subtasks: 20`, `temperature: 0.3`, `require_chain_of_thought: true`, `flag_destructive_ops: true` |
| `executor.yaml` | `max_retries`, timeout settings, tool registry binding |
| `verifier.yaml` | Assertion matching rules, soft/hard fail thresholds |
| `supervisor.yaml` | `max_auto_retries: 3`, escalation policy, retry strategy |

**System prompts** are Jinja2 templates in `prompts/`:
- `planner_system.j2` — Decomposition prompt with JSON output schema
- `executor_system.j2` — Tool calling execution prompt
- `verifier_system.j2` — Output validation prompt
- `supervisor_system.j2` — Decision-making prompt (COMPLETE/RETRY/ESCALATE)

---

### MCP Tool Registry

**Directory:** `configs/mcp/`

| File | Tools | Actions |
|------|-------|---------|
| `composio.yaml` | Global Composio settings | API config, enabled apps |
| `github.yaml` | GitHub | Create issues, list PRs, push files, create branches, get content |
| `os_automation.yaml` | Shell/OS | Execute commands, read/write files, list dirs, process management |
| `browser.yaml` | Browser | Navigate, click, screenshot, get page source, extract text |

To add MCP tools:
```bash
# Seed the registry from Composio
make seed

# Or manually:
uv run python scripts/seed_registry.py --list
```

---

## Usage

### Starting the API Server

```bash
# Development (hot reload)
make dev

# Or directly:
uv run uvicorn orion.api.server:app --reload --host 0.0.0.0 --port 8080
```

The API is now live at `http://localhost:8080`.

### Submitting a Task

```bash
curl -X POST http://localhost:8080/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Create a file /tmp/hello.txt with content hello world",
    "timeout_seconds": 120
  }'
```

Response (202 Accepted):
```json
{
  "task_id": "task_a1b2c3d4e5f6",
  "status": "QUEUED",
  "created_at": "2026-04-11T14:30:00Z",
  "estimated_seconds": 120
}
```

### Streaming Task Events

```bash
curl -N http://localhost:8080/v1/tasks/task_a1b2c3d4e5f6/stream
```

SSE output:
```
event: STEP_START
data: {"step": "pipeline_start"}

event: STEP_DONE
data: {"step": "planner", "plan": "..."}

event: STEP_DONE
data: {"step": "executor"}

event: TASK_DONE
data: {"result": "..."}
```

### Checking Task Status

```bash
curl http://localhost:8080/v1/tasks/task_a1b2c3d4e5f6
```

### Cancelling a Task

```bash
curl -X DELETE http://localhost:8080/v1/tasks/task_a1b2c3d4e5f6
```

### Browsing Tools

```bash
# List all tools
curl http://localhost:8080/v1/tools

# Get a specific tool schema
curl http://localhost:8080/v1/tools/GITHUB_CREATE_ISSUE
```

---

## API Reference

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `POST` | `/v1/tasks` | Submit a new task | `202` + TaskResponse |
| `GET` | `/v1/tasks/{id}` | Get task status & result | `200` + TaskDetailResponse |
| `GET` | `/v1/tasks/{id}/stream` | SSE stream of events | Stream of TaskEvent |
| `DELETE` | `/v1/tasks/{id}` | Cancel a running task | `204` |
| `GET` | `/v1/tools` | List all MCP tools | `200` + list of ToolSchema |
| `GET` | `/v1/tools/{name}` | Get one tool schema | `200` + ToolSchema |
| `GET` | `/health` | Liveness probe | `200` + `{"status": "ok"}` |
| `GET` | `/ready` | Readiness probe | `200` + dep check results |

**Error format** (RFC 7807):
```json
{
  "type": "/errors/task-not-found",
  "title": "Task Not Found",
  "status": 404,
  "detail": "Task 'task_xyz' not found",
  "task_id": "task_xyz"
}
```

**Rate limiting**: 429 with `Retry-After` header when `MAX_CONCURRENT_TASKS` is exceeded.

---

## Docker Deployment

```bash
# Build and start the full stack
make docker-build
make docker-up

# Or directly:
docker compose up -d
```

This starts 4 services:

| Service | Port | Description |
|---------|------|-------------|
| `orion-api` | 8080 | FastAPI application |
| `orion-qdrant` | 6333 | Qdrant vector DB |
| `orion-prometheus` | 9090 | Metrics collection |
| `orion-grafana` | 3000 | Dashboards (login: `admin` / `admin`) |

```bash
# Dev mode (hot reload via volume mount)
docker compose --profile dev up -d

# View logs
docker compose logs -f orion-api

# Health check
make health

# Stop everything
make docker-down
```

---

## Kubernetes Production Deployment

Manifests are in `infra/k8s/`:

```bash
# Apply all manifests
kubectl apply -f infra/k8s/

# This creates:
#   - Deployment (2 replicas, rolling updates)
#   - Service (ClusterIP + optional LoadBalancer)
#   - HPA (auto-scale on 70% CPU, 2-10 pods)
#   - ConfigMap (non-secret config)
```

Resource allocation per pod:

| Resource | Request | Limit |
|----------|---------|-------|
| CPU | 500m | 2000m |
| Memory | 512Mi | 2Gi |

**Secrets**: Create a K8s Secret named `orion-secrets` with your API keys:
```bash
kubectl create secret generic orion-secrets \
  --from-literal=GROQ_API_KEY=gsk_xxx \
  --from-literal=COMPOSIO_API_KEY=xxx
```

---

## Observability

### Prometheus Metrics (port 9091)

| Metric | Type | Labels |
|--------|------|--------|
| `openclaw_llm_requests_total` | Counter | provider, model, status |
| `openclaw_llm_latency_seconds` | Histogram | provider |
| `openclaw_llm_tokens_total` | Counter | provider, type (input/output) |
| `openclaw_llm_cost_usd_total` | Counter | provider |
| `openclaw_provider_status` | Gauge | provider (0=down, 1=up) |
| `openclaw_tool_calls_total` | Counter | tool, category, status |
| `openclaw_tool_latency_seconds` | Histogram | tool |
| `openclaw_tasks_total` | Counter | status (pass/fail/escalate) |
| `openclaw_task_duration_seconds` | Histogram | — |
| `openclaw_memory_working_tokens` | Gauge | agent |
| `openclaw_memory_longterm_documents_total` | Gauge | — |
| `openclaw_circuit_breaker_state` | Gauge | provider (0=closed, 1=open) |

### Grafana Dashboards (port 3000)

Pre-built dashboard with 4 rows:
1. **LLM Backend** — Provider status, request rate, latency p50/p95, cost
2. **Agent Pipeline** — Tasks/hour, success rate, duration
3. **Tool Usage** — Top-10 tools by call count, error rates
4. **Memory** — Working memory utilization, long-term document count

### Structured Logging

- **Dev mode**: Coloured console output with timestamps
- **Prod mode** (`ORION_ENV=production`): JSON to stdout
- **Secret redaction**: API keys, tokens, and passwords are automatically masked in all log output

---

## Testing

```bash
# Run all tests (176 tests)
make test

# Unit tests only
make test-unit

# Integration tests only
make test-int

# Run the evaluation suite (10 canonical tasks)
make eval
```

Test breakdown:

| Suite | Count | What It Covers |
|-------|-------|---------------|
| `tests/unit/` | ~155 | Core types, exceptions, LLM router, agents, safety, registry, memory |
| `tests/integration/` | ~21 | Full pipeline, MCP client, API endpoints, memory lifecycle |
| `tests/fixtures/` | — | Mock responses, sample tasks |

---

## Project Structure

```
orion/
├── agents/              # HiClaw 4-agent pipeline
│   ├── planner.py       #   Task decomposition → DAG
│   ├── executor.py      #   Tool execution engine
│   ├── verifier.py      #   Output validation
│   └── supervisor.py    #   Retry / escalate / complete
├── api/                 # FastAPI REST server
│   ├── server.py        #   App factory + lifespan
│   ├── schemas.py       #   Pydantic models (frozen)
│   └── routes/          #   tasks, tools, status
├── core/                # Shared primitives
│   ├── task.py          #   Task, SubTask, TaskDAG
│   ├── exceptions.py    #   Exception hierarchy (30+ types)
│   └── enums.py         #   TaskStatus, ToolCategory, etc.
├── llm/                 # Adaptive LLM routing
│   ├── router.py        #   AdaptiveLLMRouter (failover chain)
│   ├── circuit_breaker.py
│   └── providers/       #   vLLM, Groq, OpenRouter wrappers
├── memory/              # Two-tier memory
│   ├── working.py       #   In-context + eviction
│   ├── longterm.py      #   Qdrant vector store
│   ├── embedder.py      #   all-MiniLM-L6-v2 singleton
│   └── retriever.py     #   Unified retrieval facade
├── observability/       # Tracing, metrics, logging
│   ├── tracer.py        #   LangSmith / structlog fallback
│   ├── metrics.py       #   12 Prometheus metrics
│   └── logger.py        #   structlog + secret redaction
├── safety/              # Defensive execution layer
│   ├── manifest.py      #   YAML permission checking
│   ├── gate.py          #   Destructive operation gate
│   ├── sandbox.py       #   Subprocess sandboxing
│   └── rollback.py      #   LIFO checkpoint/restore
└── tools/               # Composio MCP integration
    ├── registry.py      #   ToolRegistry singleton
    ├── mcp_client.py    #   7-step invocation pipeline
    ├── selector.py      #   Semantic tool selection
    └── categories/      #   GitHub, OS, Browser, SaaS wrappers

configs/                 # All YAML configuration
├── agents/              #   Per-agent settings
├── llm/                 #   Provider + router config
├── mcp/                 #   Tool definitions
├── memory/              #   Working + longterm settings
└── safety/              #   Permissions + sandbox

infra/                   # Infrastructure
├── grafana/             #   Dashboard JSON
├── k8s/                 #   Deployment, Service, HPA
└── prometheus/          #   Alerting rules

prompts/                 #   Jinja2 system prompt templates
scripts/                 #   Dev + ops tooling
tests/                   #   Unit, integration, fixtures
```

---

## Makefile Reference

```bash
make help          # Show all targets

# ── Setup ──
make install       # Install all dependencies via uv

# ── Dev ──
make dev           # Start API with hot reload (port 8080)

# ── Test ──
make test          # Run ALL tests (unit + integration)
make test-unit     # Run unit tests only
make test-int      # Run integration tests only

# ── Quality ──
make lint          # Run ruff linter
make lint-fix      # Auto-fix lint + format
make typecheck     # Run mypy --strict

# ── Docker ──
make docker-build  # Build Docker image
make docker-up     # Start full stack (API + Qdrant + Prometheus + Grafana)
make docker-up-dev # Start with hot reload
make docker-down   # Stop everything

# ── Ops ──
make eval          # Run the 10-task evaluation suite
make health        # System health check
make seed          # Seed tool registry from Composio

# ── Cleanup ──
make clean         # Remove caches, build artifacts
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY not set` | Copy `.env.example` to `.env` and add your Groq key from [console.groq.com](https://console.groq.com) |
| `429 Too Many Tasks` | Increase `MAX_CONCURRENT_TASKS` in `.env` or wait for running tasks to complete |
| `Qdrant connection failed` | Start Qdrant via `docker compose up qdrant` or set `QDRANT_URL` correctly |
| `sentence_transformers OSError` on Windows | Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) for PyTorch DLL loading |
| `Circuit breaker tripped` | A provider had 3+ consecutive failures. It auto-recovers after 60s. Check provider status in Grafana. |
| `PermissionDeniedError` | The action is blocked by `configs/safety/permissions.yaml`. Add it to the allowed list. |
| `MaxIterationsError` | The executor hit the 20-step limit. Increase `ORION_MAX_EXECUTOR_ITERATIONS` or simplify the task. |
| Slow startup | First run downloads the `all-MiniLM-L6-v2` model (~80MB). Subsequent starts use cache. |

---

## License

MIT

