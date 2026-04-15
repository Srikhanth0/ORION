# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-04-14

### 🚀 Highlights

ORION v1.0.0 is the first production release of the hierarchical OS automation
agent. It combines a 4-agent HiClaw pipeline with open-source tooling, vision-based
computer control, and a full observability stack.

### Added

#### Core Architecture
- **AgentScope-powered 4-agent HiClaw pipeline** — Planner, Executor, Verifier, Supervisor
- **AsyncIO DAG parallel execution** — independent subtasks run concurrently via `asyncio.gather`
- **Topological sort** with cycle detection for dependency resolution

#### Tool Integration
- **Open-source MCP stdio tool registry** — GitHub, OS, browser, SaaS categories
- **Native Python tool support** — `register_native()` for tools that bypass MCP
- **Tool selector** with semantic (sentence-transformer) and keyword scoring
- **Defensive MCP client** with exponential backoff retry (3 attempts)

#### Vision & Computer Control
- **Qwen2.5-VL via Colab + ngrok** — remote vision model for screen analysis
- **pyautogui computer control** — click, type, keypress from bounding-box coords
- **5 vision tools**: `take_screenshot`, `analyze_screen`, `click_element`, `type_text`, `press_key`

#### Memory
- **ChromaDB embedded long-term memory** with sentence-transformer embeddings
- **Working memory** with token-based eviction and utilization tracking

#### Safety
- **Permission manifest** — per-category allow/deny rules
- **Destructive operation gate** — human-in-the-loop approval for risky ops
- **Rollback engine** — LIFO checkpoint/restore with 50-entry stack
- **Execution sandbox** — path validation, env sanitization, command filtering

#### Observability
- **Structured logging** via structlog — JSON in production, color console in dev
- **Prometheus metrics** — LLM latency, tool calls, task duration, vision API, circuit breaker
- **LangSmith tracing** integration
- **Full SSE streaming** — 7 fine-grained event types for real-time pipeline tracking

#### Infrastructure
- **Multi-stage Dockerfile** — < 600MB production image with vision deps
- **GitHub Actions CI** — lint, typecheck, unit tests, integration tests, container build
- **Kubernetes manifests** — Deployment, Service, HPA with custom metric scaling
- **Podman Compose** stack for local development

#### Reliability
- **3-state circuit breaker** (CLOSED → OPEN → HALF_OPEN) per LLM provider
- **Adaptive LLM router** with tiered fallback chain (vLLM → Groq → OpenRouter)
- **25-task evaluation suite** spanning OS, browser, GitHub, vision, multi-step, and error scenarios

### Infrastructure
- Podman-native container stack
- `make test`, `make eval`, `make health`, `make dev` targets
- `.env.example` with all required environment variables

### Testing
- **248 unit + integration tests** — all passing
- **25 canonical eval tasks** covering all tool categories and error cases

---

## [0.1.0] — 2026-04-11

### Added
- Initial scaffolding and project structure
- Basic agent interfaces
- Router configuration
- FastAPI server skeleton
