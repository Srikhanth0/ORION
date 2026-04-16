# ORION: Autonomous OS Automation Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-005850?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/AgentScope-7B61FF?style=for-the-badge&logo=ai&logoColor=white" alt="AgentScope">
  <img src="https://img.shields.io/badge/MCP-892CA0?style=for-the-badge&logo=mcp&logoColor=white" alt="MCP">
  <img src="https://img.shields.io/badge/ChromaDB-FF4154?style=for-the-badge&logo=chroma&logoColor=white" alt="ChromaDB">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

---

## Overview

**ORION** is a professional-grade autonomous agent designed for high-stakes OS automation. It uses a **hierarchical multi-agent pipeline** ("HiClaw") combined with **AsyncIO parallel DAG execution** to decompose instructions into dependency-aware graphs, execute them through a defensive safety shield, and verify results before completion.

ORION supports:
- Complex GitHub workflows
- Browser automation
- Shell operations
- **Vision-Language Models (VLM)** for GUI native control via PyAutoGUI

Every step is **traceable, safe, and recoverable**.

---

## Architecture

ORION follows a strict layered architecture:

```
User Instruction
      │
      ▼
FastAPI REST Server
      │
      ▼
Adaptive LLM Router (vLLM → Groq → OpenRouter)
      │
      ├──────────────────────┬─────────────────────┐
      ▼                     ▼                     ▼
┌─────────┐          ┌──────────┐          ┌─────────┐
│ Planner │ ──────▶ │ Executor│ ──────▶ │ Verifier│
│ Agent  │          │ (Async   │          │ Agent   │
│        │          │   DAG)  │          │         │
└─────────┘          └──────────┘          └─────────┘
      │                                   │
      ▼                                   ▼
┌─────────────────────┐           ┌────────────┐
│ Supervisor Agent   │ ◀──────── │ Retry/Fix │
│ (decides next step) │           │  Loop    │
└─────────────────────┘           └────────────┘
```

### HiClaw Pipeline
1. **PlannerAgent** - Decomposes user requests into a JSON-based DAG structure
2. **ExecutorAgent** - Executes subtasks asynchronously using `asyncio.gather` for parallel execution
3. **VerifierAgent** - Validates execution results
4. **SupervisorAgent** - Decides: COMPLETE, RETRY, ROLLBACK, or ESCALATE

### Safety Shield
- **Permission Manifests** (`configs/safety/permissions.yaml`) - Prevents unauthorized tools
- **Destructive Op Gate** - Forces Human-In-The-Loop approval for dangerous operations
- **LIFO Rollback Engine** - Checkpoints file edits for recovery

### Memory
- **Working Memory** - In-context task state
- **Long-Term Memory** - ChromaDB vector store for semantic retrieval

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Orchestration | AgentScope |
| API Server | FastAPI, Uvicorn |
| LLM Integration | OpenAI, Groq, httpx |
| Tool Integration | MCP (Model Context Protocol) |
| Vision/GUI | PyAutoGUI, Pillow |
| Memory | ChromaDB, sentence-transformers |
| CLI | Rich, prompt-toolkit |
| Observability | LangSmith, Prometheus, structlog |

---

## File Structure

```
ORION/
├── .agents/                  # Specialized agent behaviors
├── configs/                  # YAML configuration files
│   ├── agents/              # Agent configurations
│   ├── llm/                # LLM provider configs
│   ├── memory/             # Memory configs
│   ├── mcp/               # MCP server configs
│   └── safety/            # Safety configurations
├── orion/                   # Core source code
│   ├── agents/             # HiClaw agents (Planner, Executor, Verifier, Supervisor)
│   ├── api/              # FastAPI routes
│   ├── core/             # Core primitives (Task, Result, Exceptions)
│   ├── llm/              # LLM providers & routing
│   ├── memory/           # Working & long-term memory
│   ├── observability/     # Logging, metrics, tracing
│   ├── orchestrator/     # Pipeline & DAG dispatcher
│   ├── safety/          # Safety mechanisms
│   └── tools/           # Tool registry & categories
├── scripts/                 # Utility scripts
├── tests/                   # Unit & integration tests
├── pyproject.toml           # Project dependencies
├── Makefile               # Development tasks
└── orion_cli.py           # CLI entry point
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/tasks` | Submit a task |
| `GET` | `/v1/tasks/{id}` | Get task status & results |
| `GET` | `/v1/tasks/{id}/stream` | SSE stream of task events |
| `GET` | `/v1/tools` | List available tools |
| `POST` | `/v1/tasks/{id}/rollback` | Rollback to a checkpoint |

---

## CLI Usage

```bash
# Start interactive REPL
uv run orion

# List available tools
orion --tools

# Submit a task
orion --submit "Create a new file called test.txt"

# Run in sandbox mode
orion --safe-mode --submit "Delete all files"
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `<instruction>` | Submit a task to the agent |
| `/stream <task_id>` | Stream live events from a task |
| `/tools` | List all available tools |
| `/safe-mode` | Toggle Podman sandbox mode |
| `/help` | Show help message |
| `/quit` | Exit the CLI |

---

## Development

```bash
# Install dependencies
make install

# Start development server
make dev

# Run tests
make test

# Run evaluation suite
make eval

# Health check
make health
```

---

## License

MIT License - See LICENSE for details.
