# OpenClaw — Project File Structure

```
openclaw/
│
├── README.md
├── pyproject.toml                    # uv / poetry project manifest
├── .env.example                      # env vars template (API keys, ports)
├── .env                              # local secrets (gitignored)
├── docker-compose.yml                # full stack: vLLM + Qdrant + Prometheus
├── Makefile                          # dev shortcuts: make run, make test, etc.
│
├── configs/                          # all YAML config — no hardcoding
│   ├── agents/
│   │   ├── planner.yaml             # Planner agent: model, memory, prompt ref
│   │   ├── executor.yaml            # Executor agent config
│   │   ├── verifier.yaml            # Verifier agent config
│   │   └── supervisor.yaml          # Supervisor agent + HITL settings
│   ├── llm/
│   │   ├── router.yaml              # Fallback chain: vLLM → Groq → OpenRouter
│   │   ├── vllm.yaml                # Qwen 2.5 vLLM endpoint + params
│   │   ├── groq.yaml                # Groq model + rate-limit thresholds
│   │   └── openrouter.yaml          # OpenRouter model + budget cap
│   ├── mcp/
│   │   ├── composio.yaml            # Composio API key + enabled tool list
│   │   ├── github.yaml              # GitHub MCP tool config + permissions
│   │   ├── os_automation.yaml       # Shell / FS tool allowlists
│   │   └── browser.yaml             # Playwright MCP settings
│   ├── memory/
│   │   ├── working.yaml             # In-context window size + summarisation
│   │   └── longterm.yaml            # Qdrant / ChromaDB connection + collection
│   └── safety/
│       ├── permissions.yaml         # Per-tool permission manifest
│       └── sandbox.yaml             # Exec timeout, resource caps, rollback
│
├── openclaw/                         # main Python package
│   ├── __init__.py
│   │
│   ├── core/                         # Framework-agnostic primitives
│   │   ├── __init__.py
│   │   ├── task.py                  # Task, Subtask, TaskDAG dataclasses
│   │   ├── message.py               # OpenClaw message schema (extends AgentScope Msg)
│   │   ├── result.py                # StepResult, TaskResult, RollbackPoint
│   │   └── exceptions.py            # ToolError, PlanError, SafetyError, etc.
│   │
│   ├── llm/                          # LLM router + provider wrappers
│   │   ├── __init__.py
│   │   ├── router.py                # AdaptiveLLMRouter: health check + fallback
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── vllm_provider.py     # Qwen 2.5 via vLLM (OpenAI-compat)
│   │   │   ├── groq_provider.py     # Groq client + quota tracking
│   │   │   └── openrouter_provider.py # OpenRouter client + budget guard
│   │   └── health.py                # Async health-check loop for each provider
│   │
│   ├── orchestrator/                 # AgentScope pipeline wrappers
│   │   ├── __init__.py
│   │   ├── pipeline.py              # OpenClawPipeline: builds AgentScope pipelines
│   │   ├── dispatcher.py            # Routes incoming tasks to correct agent chain
│   │   └── rpc_server.py            # AgentScope RPC server entry point
│   │
│   ├── agents/                       # HiClaw agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseOpenClawAgent(agentscope.Agent)
│   │   ├── planner.py               # PlannerAgent: ReAct + CoT → TaskDAG
│   │   ├── executor.py              # ExecutorAgent: tool-use loop + retry
│   │   ├── verifier.py              # VerifierAgent: assertions + self-critique
│   │   └── supervisor.py            # SupervisorAgent: HITL + abort/rollback
│   │
│   ├── memory/                       # Two-tier memory system
│   │   ├── __init__.py
│   │   ├── working.py               # WorkingMemory: AgentScope in-context store
│   │   ├── longterm.py              # LongTermMemory: Qdrant/ChromaDB client
│   │   ├── embedder.py              # Task embedding (sentence-transformers)
│   │   └── retriever.py             # RAG retrieval for past execution plans
│   │
│   ├── tools/                        # Composio MCP integration layer
│   │   ├── __init__.py
│   │   ├── registry.py              # ToolRegistry: index, schema, capability score
│   │   ├── mcp_client.py            # Composio MCP client + tool executor
│   │   ├── selector.py              # ToolSelector: scores tools against subtask
│   │   └── categories/
│   │       ├── __init__.py
│   │       ├── github_tools.py      # GitHub MCP wrappers
│   │       ├── os_tools.py          # Shell / FS / process tools
│   │       ├── browser_tools.py     # Playwright MCP wrappers
│   │       └── saas_tools.py        # Slack, Notion, Linear, Gmail wrappers
│   │
│   ├── safety/                       # Sandboxing + permission enforcement
│   │   ├── __init__.py
│   │   ├── manifest.py              # PermissionManifest loader + checker
│   │   ├── sandbox.py               # ExecSandbox: resource limits + timeout
│   │   ├── rollback.py              # RollbackEngine: checkpoint + restore
│   │   └── gate.py                  # DestructiveOpGate: requires supervisor OK
│   │
│   ├── observability/                # Logging, tracing, metrics
│   │   ├── __init__.py
│   │   ├── tracer.py                # LangSmith / Phoenix trace export
│   │   ├── metrics.py               # Prometheus counters: tokens, latency, cost
│   │   └── logger.py                # Structured JSON logger with task IDs
│   │
│   └── api/                          # External-facing API surface
│       ├── __init__.py
│       ├── server.py                # FastAPI app factory
│       ├── routes/
│       │   ├── tasks.py             # POST /tasks, GET /tasks/{id}
│       │   ├── status.py            # GET /status, GET /health
│       │   └── tools.py             # GET /tools — list available MCP tools
│       └── schemas.py               # Pydantic request/response models
│
├── prompts/                          # All system prompts (versioned Jinja2)
│   ├── planner_system.j2            # Planner: ReAct + DAG instructions
│   ├── executor_system.j2           # Executor: tool-use loop format
│   ├── verifier_system.j2           # Verifier: assertion + critique format
│   └── supervisor_system.j2         # Supervisor: escalation decision format
│
├── scripts/                          # Dev + ops utilities
│   ├── start_vllm.sh                # Launch Qwen 2.5 via vLLM
│   ├── seed_registry.py             # Pre-populate tool registry from Composio
│   ├── healthcheck.py               # End-to-end smoke test
│   └── eval_task.py                 # Run a single task + print trace
│
├── tests/
│   ├── unit/
│   │   ├── test_router.py           # LLM router fallback logic
│   │   ├── test_planner.py          # DAG generation from prompts
│   │   ├── test_executor.py         # Tool invocation + retry
│   │   ├── test_verifier.py         # Assertion checks
│   │   ├── test_registry.py         # Tool registry + selector
│   │   └── test_safety.py           # Permission manifest + gate
│   ├── integration/
│   │   ├── test_pipeline.py         # End-to-end agent pipeline
│   │   ├── test_composio_mcp.py     # Live MCP tool calls (mocked)
│   │   └── test_memory.py           # Working + long-term memory round-trip
│   └── fixtures/
│       ├── sample_tasks.json        # Reference task inputs for tests
│       └── mock_tool_responses.json # Composio MCP mock payloads
│
├── docs/
│   ├── architecture.md              # This diagram + narrative
│   ├── setup.md                     # Local dev quickstart
│   ├── tools_reference.md           # All 40+ Composio MCP tools
│   └── adding_agents.md             # How to create a custom HiClaw agent
│
└── infra/
    ├── k8s/                          # Kubernetes manifests (optional)
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   └── hpa.yaml                 # Horizontal pod autoscaler
    ├── prometheus/
    │   └── openclaw_rules.yaml      # Alert rules for token cost + error rate
    └── grafana/
        └── dashboard.json           # Pre-built Grafana dashboard
```

## Key dependency map

| Module            | Depends on                                          |
|-------------------|-----------------------------------------------------|
| `agents/`         | `llm/router`, `memory/`, `tools/registry`, `safety/`|
| `orchestrator/`   | `agents/`, `core/task`, `observability/`            |
| `tools/`          | `safety/manifest`, `observability/tracer`           |
| `llm/router`      | `llm/providers/*`, `llm/health`                     |
| `api/`            | `orchestrator/`, `core/schemas`                     |
| `safety/`         | `configs/safety/`, `core/exceptions`                |
