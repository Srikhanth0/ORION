# ORION — Professional Agent Harness: Complete Implementation Plan

> **Problem Statement**: The current ORION harness with Qwen 3 vLLM 4B treats every prompt as a
> file-write operation. There is no planning phase, no human review, no tool diversity enforcement,
> and no task gating. This document is the canonical blueprint to fix all of that.

---

## Table of Contents

1. [Root Cause Analysis](#1-root-cause-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Phase 0 — Foundation Hardening](#phase-0--foundation-hardening)
4. [Phase 1 — Planning Layer (PlannerAgent + HiTL)](#phase-1--planning-layer)
5. [Phase 2 — Task File Derivation](#phase-2--task-file-derivation)
6. [Phase 3 — ToolGuard & Specialist Dispatcher](#phase-3--toolguard--specialist-dispatcher)
7. [Phase 4 — Checklist-Gated Executor](#phase-4--checklist-gated-executor)
8. [Phase 5 — VerifierAgent & Rollback](#phase-5--verifieragent--rollback)
9. [Phase 6 — AgentScope Pipeline Wiring](#phase-6--agentscope-pipeline-wiring)
10. [Phase 7 — Observability & CLI Polish](#phase-7--observability--cli-polish)
11. [Data Contracts & Schemas](#data-contracts--schemas)
12. [File Diff Map](#file-diff-map)
13. [Risk Register](#risk-register)
14. [Testing Strategy](#testing-strategy)

---

## 1. Root Cause Analysis

### 1.1 Why the Agent Only Uses the Write Tool

| Symptom | Underlying Cause |
|---------|-----------------|
| Every prompt → file write | No structured planning step; LLM defaults to the first registered tool |
| Repeated tool calls | No tool-use tracking per session/task |
| No goal decomposition | PlannerAgent exists in structure but is not called before execution |
| No HiTL | `UserAgent` from AgentScope is never instantiated |
| Tasks run serially without gating | No checklist structure; next task fires immediately |
| Qwen 4B confuses intent | System prompt is too vague; no explicit output schema enforced |

### 1.2 Core Design Gaps

```
Current Flow (broken):
  User Prompt → ExecutorAgent → write_file() → Done

Required Flow:
  User Prompt
    → PlannerAgent  (produces structured DAG plan as JSON)
    → HumanReviewAgent  (AgentScope UserAgent — approve / edit / reject)
    → TaskDeriver  (persists approved plan as tasks.json + checklist.json)
    → ChecklistGate  (blocks next task until current is ✅)
    → ToolGuardDispatcher  (per-task allowed-tool manifest; rejects wrong calls)
    → SpecialistExecutor  (one focused executor per task type)
    → VerifierAgent  (validates success criteria)
    → SupervisorAgent  (COMPLETE | RETRY | ROLLBACK | ESCALATE)
```

---

## 2. Target Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ORION Harness                        │
│                                                              │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │  FastAPI │───▶│  Orchestrator    │───▶│  AgentScope   │ │
│  │  /v1/    │    │  (pipeline.py)   │    │  Pipeline     │ │
│  └──────────┘    └────────┬─────────┘    └───────┬───────┘ │
│                           │                       │         │
│              ┌────────────▼────────────┐          │         │
│              │     PlannerAgent        │          │         │
│              │  (Qwen → structured     │          │         │
│              │   JSON DAG plan)        │          │         │
│              └────────────┬────────────┘          │         │
│                           │ plan.json             │         │
│              ┌────────────▼────────────┐          │         │
│              │   HumanReviewAgent      │◀─────────┘         │
│              │  (AgentScope UserAgent) │                     │
│              │  approve/edit/reject    │                     │
│              └────────────┬────────────┘                     │
│                           │ approved_plan.json               │
│              ┌────────────▼────────────┐                     │
│              │     TaskDeriver         │                     │
│              │  tasks.json             │                     │
│              │  checklist.json         │                     │
│              └────────────┬────────────┘                     │
│                           │                                  │
│              ┌────────────▼────────────┐                     │
│              │   ChecklistGate         │                     │
│              │  (blocks until ✅)       │                     │
│              └────────────┬────────────┘                     │
│                           │ task_n                           │
│              ┌────────────▼────────────┐                     │
│              │   ToolGuardDispatcher   │                     │
│              │  allowed_tools per task │                     │
│              │  dedup + reject guard   │                     │
│              └────────────┬────────────┘                     │
│                     ┌─────┴──────┐                           │
│            ┌────────▼──┐  ┌──────▼───────┐                  │
│            │  Shell    │  │  GitHub      │  ... specialists  │
│            │ Executor  │  │  Executor    │                   │
│            └────────┬──┘  └──────┬───────┘                  │
│                     └─────┬──────┘                           │
│              ┌────────────▼────────────┐                     │
│              │    VerifierAgent        │                     │
│              │  checks success criteria│                     │
│              └────────────┬────────────┘                     │
│                           │                                  │
│              ┌────────────▼────────────┐                     │
│              │    SupervisorAgent      │                     │
│              │  COMPLETE/RETRY/        │                     │
│              │  ROLLBACK/ESCALATE      │                     │
│              └─────────────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

---

## Phase 0 — Foundation Hardening

**Goal**: Fix the Qwen 4B vLLM integration so the model produces structured outputs reliably.

### 0.1 — Structured Output Enforcement via vLLM Grammar

**File**: `orion/llm/vllm_provider.py`

- Set `guided_json` or `response_format={"type": "json_object"}` on every vLLM call.
- Always pass a Pydantic schema as the grammar constraint so Qwen cannot produce free text.
- Use `temperature=0.1` for planning calls (deterministic), `temperature=0.4` for creative/code tasks.

```python
# orion/llm/vllm_provider.py

class VLLMProvider:
    async def complete_structured(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        temperature: float = 0.1,
    ) -> BaseModel:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            response_format={
                "type": "json_object",
                "schema": schema.model_json_schema(),
            },
        )
        raw = response.choices[0].message.content
        return schema.model_validate_json(raw)
```

### 0.2 — System Prompt Overhaul

**File**: `configs/agents/planner.yaml`

The system prompt must:
1. Define the agent's SOLE responsibility (plan decomposition — nothing else).
2. Specify the EXACT JSON output schema.
3. List all valid tool names so the model knows what tools exist.
4. Forbid free-text preamble: `"Your ONLY output is a valid JSON object matching the schema."`

### 0.3 — Fallback Chain Fix

**File**: `orion/llm/router.py`

- Add exponential backoff before failover (vLLM → Groq → OpenRouter).
- Log which provider was used for each call to detect degraded routing.
- Return a structured `LLMResponse(provider, latency_ms, tokens_used, content)` so supervisors can flag when fallback is active.

### 0.4 — Tool Registry Audit

**File**: `orion/tools/registry.py`

- Enumerate ALL registered tools with: `name`, `description`, `input_schema`, `category`, `is_destructive`, `idempotent`.
- On startup, print the registry summary to stdout so operators can verify what is loaded.
- Remove or disable the bare `write_file` shortcut that is currently matching everything.

**Checklist for Phase 0**:
- [ ] `vllm_provider.py` uses `guided_json` for all structured calls
- [ ] `planner.yaml` system prompt specifies exact JSON schema inline
- [ ] `router.py` logs active provider per call
- [ ] `registry.py` exports full tool manifest on startup
- [ ] Unit test: Qwen returns valid JSON for 10 diverse prompts without free-text leakage

---

## Phase 1 — Planning Layer

**Goal**: Replace the ad-hoc "just do it" executor with a mandatory two-step plan → HiTL approval flow.

### 1.1 — PlannerAgent

**File**: `orion/agents/planner_agent.py`

**Responsibilities**:
- Accept a raw user prompt as input.
- Call `vllm_provider.complete_structured(schema=ImplementationPlan)`.
- Return a validated `ImplementationPlan` Pydantic object.
- Store the raw plan JSON to `~/.orion/plans/{task_id}_plan.json`.

**ImplementationPlan Schema** (see §Data Contracts):
```
ImplementationPlan
  ├── task_id: str (UUID4)
  ├── original_prompt: str
  ├── goal_summary: str (≤ 100 chars)
  ├── estimated_complexity: LOW | MEDIUM | HIGH
  ├── tasks: list[TaskNode]
  │     ├── id: str (T-001, T-002 …)
  │     ├── title: str
  │     ├── description: str
  │     ├── category: SHELL | FILE | GIT | BROWSER | VISION | API | MEMORY
  │     ├── allowed_tools: list[str]  ← EXPLICIT, not inferred
  │     ├── prerequisites: list[str]  ← task IDs that must be ✅ first
  │     ├── success_criteria: list[str]
  │     ├── estimated_duration_sec: int
  │     └── is_destructive: bool
  └── rollback_strategy: str
```

**Planner System Prompt Key Rules**:
```
You are ORION PlannerAgent. You receive a user instruction and produce ONLY a
valid JSON ImplementationPlan. Rules:
1. Decompose into 2–10 discrete tasks. Never produce a single mega-task.
2. Each task must specify exactly which tools are allowed (from the registry).
3. Mark any task that deletes, overwrites, or modifies existing resources as is_destructive=true.
4. Prerequisites must form a valid DAG (no cycles).
5. Do not execute anything. Plan only.
6. Output ONLY the JSON. No preamble, no explanation.
```

### 1.2 — HumanReviewAgent (AgentScope UserAgent)

**File**: `orion/agents/human_review_agent.py`

This is the Human-In-The-Loop gate. Uses `agentscope.agents.UserAgent`.

**Display Format** (rendered via `rich`):
```
╔══════════════════════════════════════════════════════╗
║          ORION — Implementation Plan Review          ║
╠══════════════════════════════════════════════════════╣
║  Task ID : {task_id}                                 ║
║  Goal    : {goal_summary}                            ║
║  Complexity: {estimated_complexity}                  ║
╠══════════════════════════════════════════════════════╣
║  TASKS                                               ║
║  ─────────────────────────────────────────────────   ║
║  [T-001] {title}                                     ║
║    Category  : {category}                            ║
║    Tools     : {allowed_tools}                       ║
║    Depends on: {prerequisites}                       ║
║    Criteria  : {success_criteria}                    ║
║    Destructive: {is_destructive}                     ║
║  ...                                                 ║
╠══════════════════════════════════════════════════════╣
║  Options:                                            ║
║  [A] Approve and execute                             ║
║  [E] Edit plan (opens JSON in $EDITOR)               ║
║  [R] Reject and re-plan with feedback                ║
║  [X] Abort                                           ║
╚══════════════════════════════════════════════════════╝
```

**HumanReviewResult**:
```python
class HumanReviewResult(BaseModel):
    decision: Literal["APPROVE", "EDIT", "REJECT", "ABORT"]
    feedback: str | None  # filled on REJECT; used to re-plan
    edited_plan: ImplementationPlan | None  # filled on EDIT
```

**Re-Plan Loop**:
- On `REJECT`: feedback is appended to planner messages as a new user turn and PlannerAgent re-runs (max 3 cycles before hard abort).
- On `EDIT`: human edits the JSON file directly; system re-validates schema before accepting.
- On `ABORT`: raise `OrionAbortError`; cleanup and exit gracefully.

### 1.3 — AgentScope Pipeline Registration

**File**: `orion/orchestrator/pipeline.py`

```python
import agentscope
from agentscope.agents import UserAgent
from agentscope.pipelines import SequentialPipeline

agentscope.init(model_configs="configs/llm/agentscope_models.json")

planner = PlannerAgent(name="planner", ...)
human   = UserAgent(name="human")
deriver = TaskDeriverAgent(name="deriver", ...)

planning_pipeline = SequentialPipeline([planner, human, deriver])
```

**Checklist for Phase 1**:
- [ ] `PlannerAgent` produces valid `ImplementationPlan` JSON for 5 diverse prompts
- [ ] `HumanReviewAgent` renders rich panel and waits for input
- [ ] APPROVE path routes to TaskDeriver
- [ ] REJECT path re-invokes PlannerAgent with feedback (max 3 cycles)
- [ ] EDIT path validates edited JSON before accepting
- [ ] ABORT path raises `OrionAbortError` and cleans up
- [ ] AgentScope `UserAgent` is correctly wired (not a mock)
- [ ] `planning_pipeline` runs end-to-end in integration test

---

## Phase 2 — Task File Derivation

**Goal**: Persist the approved plan as machine-readable task and checklist files that drive all downstream execution.

### 2.1 — TaskDeriverAgent

**File**: `orion/agents/task_deriver.py`

**Input**: `ImplementationPlan` (approved)
**Outputs**:
- `~/.orion/tasks/{task_id}/tasks.json` — full task graph
- `~/.orion/tasks/{task_id}/checklist.json` — flat ordered checklist
- `~/.orion/tasks/{task_id}/tool_manifests/{T-00N}.json` — per-task tool allowlist

**`tasks.json` structure**:
```json
{
  "task_id": "abc-123",
  "status": "PENDING",
  "created_at": "2025-04-16T10:00:00Z",
  "tasks": [
    {
      "id": "T-001",
      "title": "Clone repository",
      "category": "GIT",
      "allowed_tools": ["git_clone", "fs_mkdir"],
      "prerequisites": [],
      "success_criteria": [
        "Directory ./repo exists",
        "git log returns at least 1 commit"
      ],
      "is_destructive": false,
      "status": "PENDING",
      "started_at": null,
      "completed_at": null,
      "attempts": 0,
      "result": null
    }
  ]
}
```

**`checklist.json` structure** (ordered by execution sequence after topological sort):
```json
{
  "task_id": "abc-123",
  "items": [
    { "id": "T-001", "title": "Clone repository",    "status": "PENDING" },
    { "id": "T-002", "title": "Install dependencies", "status": "PENDING" },
    { "id": "T-003", "title": "Run tests",            "status": "PENDING" }
  ]
}
```

### 2.2 — DAG Topological Sort

**File**: `orion/orchestrator/dag.py`

- Use Kahn's algorithm to topologically sort tasks by prerequisites.
- Detect cycles; raise `OrionPlanCycleError` if found.
- Expose `get_ready_tasks(checklist) → list[TaskNode]` (tasks with all prerequisites ✅).

### 2.3 — Per-Task Tool Manifests

**File**: `orion/tools/manifest.py`

For each task, write:
```json
{
  "task_id": "T-001",
  "allowed_tools": ["git_clone", "fs_mkdir"],
  "forbidden_tools": ["write_file", "shell_exec", "browser_*"],
  "used_tools": [],
  "tool_call_count": {}
}
```

- `forbidden_tools` is the complement of `allowed_tools` within the full registry.
- `used_tools` is mutated at runtime to track already-invoked tools.

**Checklist for Phase 2**:
- [ ] `TaskDeriverAgent` writes `tasks.json` and `checklist.json` atomically
- [ ] Topological sort correctly orders tasks in all test cases
- [ ] Cycle detection raises correct error
- [ ] Per-task tool manifests written for every task
- [ ] `get_ready_tasks()` correctly returns only tasks whose prerequisites are ✅
- [ ] File writes are atomic (write to tmp → rename)

---

## Phase 3 — ToolGuard & Specialist Dispatcher

**Goal**: Guarantee that each task only ever calls the tools it declared, and that no tool is
called a second time if it already succeeded.

### 3.1 — ToolGuard

**File**: `orion/safety/tool_guard.py`

```python
class ToolGuard:
    def __init__(self, manifest: TaskToolManifest):
        self.manifest = manifest

    def check(self, tool_name: str) -> ToolGuardResult:
        if tool_name in self.manifest.forbidden_tools:
            return ToolGuardResult(
                allowed=False,
                reason=f"Tool '{tool_name}' is NOT in the allowed list for task {self.manifest.task_id}. "
                       f"Allowed: {self.manifest.allowed_tools}"
            )
        if tool_name in self.manifest.used_tools:
            return ToolGuardResult(
                allowed=False,
                reason=f"Tool '{tool_name}' was already used and succeeded. "
                       f"Re-use is prohibited to prevent duplication."
            )
        return ToolGuardResult(allowed=True, reason="OK")

    def record_use(self, tool_name: str):
        self.manifest.used_tools.append(tool_name)
        self.manifest.tool_call_count[tool_name] = (
            self.manifest.tool_call_count.get(tool_name, 0) + 1
        )
        self._persist()
```

**Key behaviors**:
- Before ANY tool call, `ToolGuard.check()` must pass or the call is blocked and an error is injected into the LLM context: `"Tool X is forbidden for this task. Please use one of: [allowed list]."`.
- After a successful tool call, `ToolGuard.record_use()` is called.
- If a tool fails and needs retry, the used-tools record for that tool is NOT added (only record on success).

### 3.2 — Specialist Executor Dispatch

**File**: `orion/executor/dispatcher.py`

Rather than one monolithic executor, route each task to a category-specialized executor:

| Category | Specialist Class | Key Tools |
|----------|-----------------|-----------|
| `SHELL` | `ShellExecutor` | `shell_exec`, `shell_pipe`, `env_set` |
| `FILE` | `FileExecutor` | `fs_read`, `fs_write`, `fs_mkdir`, `fs_delete` |
| `GIT` | `GitExecutor` | `git_clone`, `git_commit`, `git_push`, `git_pr` |
| `BROWSER` | `BrowserExecutor` | `browser_navigate`, `browser_click`, `browser_fill` |
| `VISION` | `VisionExecutor` | `vlm_screenshot`, `vlm_click`, `pyautogui_type` |
| `API` | `APIExecutor` | `http_get`, `http_post`, `http_put` |
| `MEMORY` | `MemoryExecutor` | `chroma_store`, `chroma_query` |

**Dispatch Logic**:
```python
class SpecialistDispatcher:
    _registry: dict[TaskCategory, type[BaseSpecialistExecutor]] = {
        TaskCategory.SHELL: ShellExecutor,
        TaskCategory.FILE:  FileExecutor,
        TaskCategory.GIT:   GitExecutor,
        # ...
    }

    async def dispatch(self, task: TaskNode, tool_guard: ToolGuard) -> TaskResult:
        executor_cls = self._registry[task.category]
        executor = executor_cls(task=task, tool_guard=tool_guard, llm=self.llm)
        return await executor.run()
```

### 3.3 — Specialist Executor Base

**File**: `orion/executor/base_specialist.py`

```python
class BaseSpecialistExecutor(ABC):
    """
    Each specialist receives:
    - task: TaskNode (with description, success_criteria, allowed_tools)
    - tool_guard: ToolGuard (enforces allowed tools + dedup)
    - llm: LLMProvider (for reasoning about which tool to call next)
    """

    async def run(self) -> TaskResult:
        context = self._build_context()
        for step in range(self.MAX_STEPS):
            tool_call = await self.llm.decide_tool(context, self.task.allowed_tools)
            guard_result = self.tool_guard.check(tool_call.name)
            if not guard_result.allowed:
                context.inject_error(guard_result.reason)
                continue
            result = await self._execute_tool(tool_call)
            self.tool_guard.record_use(tool_call.name)
            context.append_result(result)
            if await self._is_task_done(context):
                break
        return self._build_task_result(context)
```

**Checklist for Phase 3**:
- [ ] `ToolGuard.check()` blocks forbidden tools and returns descriptive error
- [ ] `ToolGuard.check()` blocks already-used tools (dedup)
- [ ] Blocked tool error is injected into LLM context (not silently dropped)
- [ ] `SpecialistDispatcher` routes all 7 categories correctly
- [ ] Each specialist only imports tools from its own category
- [ ] Unit tests: `ToolGuard` blocks 10 invalid scenarios correctly
- [ ] Integration test: a task with `allowed_tools: ["git_clone"]` never calls `write_file`

---

## Phase 4 — Checklist-Gated Executor

**Goal**: Make task N+1 physically impossible to start until task N is marked ✅ in the checklist.

### 4.1 — ChecklistGate

**File**: `orion/orchestrator/checklist_gate.py`

```python
class ChecklistGate:
    def __init__(self, checklist_path: Path):
        self.path = checklist_path

    def get_next_task(self) -> TaskNode | None:
        """Returns the next PENDING task whose prerequisites are all DONE."""
        checklist = self._load()
        ready = [
            item for item in checklist.items
            if item.status == TaskStatus.PENDING
            and self._prerequisites_met(item, checklist)
        ]
        return ready[0] if ready else None

    def mark_done(self, task_id: str):
        checklist = self._load()
        for item in checklist.items:
            if item.id == task_id:
                item.status = TaskStatus.DONE
                item.completed_at = datetime.utcnow().isoformat()
        self._save(checklist)

    def mark_failed(self, task_id: str, reason: str):
        # Update status to FAILED; gate will block further execution
        ...

    def _prerequisites_met(self, item: ChecklistItem, checklist: Checklist) -> bool:
        prereq_ids = self._get_task_prerequisites(item.id)
        return all(
            any(ci.id == pid and ci.status == TaskStatus.DONE
                for ci in checklist.items)
            for pid in prereq_ids
        )
```

### 4.2 — Execution Loop

**File**: `orion/orchestrator/execution_loop.py`

```python
async def run_execution_loop(task_id: str):
    gate     = ChecklistGate(checklist_path(task_id))
    deriver  = TaskDeriver.load(task_id)
    supervisor = SupervisorAgent(...)

    while True:
        next_task = gate.get_next_task()

        if next_task is None:
            if gate.all_done():
                await supervisor.complete(task_id)
                break
            elif gate.has_failed():
                await supervisor.handle_failure(task_id)
                break
            else:
                # Still have PENDING tasks but none are ready → dependency deadlock
                raise OrionDeadlockError(task_id)

        # === GATE: only proceed if all prerequisites are done ===
        tool_manifest = load_tool_manifest(task_id, next_task.id)
        tool_guard    = ToolGuard(tool_manifest)
        result        = await dispatcher.dispatch(next_task, tool_guard)

        if result.success:
            gate.mark_done(next_task.id)
            await display_checklist(gate)  # live rich table update
        else:
            decision = await supervisor.decide(next_task, result)
            if decision == SupervisorDecision.RETRY:
                pass  # loop continues, same task attempted again
            elif decision == SupervisorDecision.ROLLBACK:
                await rollback_engine.rollback(task_id, up_to=next_task.id)
                gate.mark_failed(next_task.id, result.error)
                break
            elif decision == SupervisorDecision.ESCALATE:
                await human_review_agent.escalate(next_task, result)
                break
```

### 4.3 — Live Checklist Display

**File**: `orion/cli/checklist_display.py`

Using `rich.live` and `rich.table`:

```
┌────────────────────────────────────────────────────────┐
│  ORION Task Execution — abc-123                        │
├──────┬──────────────────────────────┬──────────────────┤
│  ID  │  Task                        │  Status          │
├──────┼──────────────────────────────┼──────────────────┤
│ T-001│  Clone repository            │  ✅ DONE          │
│ T-002│  Install dependencies        │  🔄 RUNNING       │
│ T-003│  Run tests                   │  ⏳ PENDING       │
│ T-004│  Deploy to staging           │  🔒 LOCKED        │
└──────┴──────────────────────────────┴──────────────────┘
```

Status icons: ✅ DONE | 🔄 RUNNING | ⏳ PENDING | ❌ FAILED | 🔒 LOCKED (prereqs not met)

**Checklist for Phase 4**:
- [ ] `ChecklistGate.get_next_task()` never returns a task with incomplete prerequisites
- [ ] Marking a task DONE immediately unlocks dependent tasks
- [ ] A FAILED task blocks all downstream dependents
- [ ] Deadlock detection raises `OrionDeadlockError`
- [ ] Live rich table updates every 500ms during execution
- [ ] `all_done()` returns true only when ALL checklist items are DONE
- [ ] Integration test: 5-task DAG executes in correct order

---

## Phase 5 — VerifierAgent & Rollback

**Goal**: Every task must pass machine-readable success criteria before being marked DONE.
Failed tasks trigger automatic rollback of their side-effects.

### 5.1 — VerifierAgent

**File**: `orion/agents/verifier_agent.py`

The verifier receives:
- `task: TaskNode` (with `success_criteria: list[str]`)
- `result: TaskResult` (stdout, files created, return code, etc.)

It produces a `VerificationReport`:
```python
class VerificationReport(BaseModel):
    task_id: str
    passed: bool
    criteria_results: list[CriterionResult]  # per-criterion pass/fail
    evidence: str  # what was checked and how
    confidence: float  # 0.0–1.0
```

**Verifier System Prompt**:
```
You are ORION VerifierAgent. Given a task's success criteria and execution result,
determine whether each criterion was met. Check file existence, process output,
exit codes, and any measurable outcomes. Return ONLY a JSON VerificationReport.
Be strict — partial completion is a FAIL.
```

**Automatic Verification Checks** (before LLM):
- Exit code == 0 for shell tasks
- File exists + non-empty for file creation tasks
- HTTP 2xx for API tasks
- Git log entry for git tasks

### 5.2 — LIFO Rollback Engine

**File**: `orion/safety/rollback_engine.py`

**Checkpointing**:
- Before any destructive operation, checkpoint the current state.
- For file edits: save original file to `~/.orion/checkpoints/{task_id}/{T-00N}/{filename}.bak`.
- For shell: record working directory state (ls -la) pre-execution.
- For git: record current HEAD SHA.

**Rollback**:
```python
class RollbackEngine:
    async def rollback(self, task_id: str, up_to: str):
        """Roll back all tasks from current back to `up_to` (LIFO order)."""
        checkpoints = self._load_checkpoints(task_id)
        # Reverse order (LIFO)
        for checkpoint in reversed(checkpoints):
            if checkpoint.task_id == up_to:
                break
            await self._apply_rollback(checkpoint)
```

**Checklist for Phase 5**:
- [ ] `VerifierAgent` checks all success criteria before marking DONE
- [ ] Automatic checks (exit code, file existence) run before LLM verification
- [ ] A failed criterion causes the task to be marked FAILED (not DONE)
- [ ] Checkpoint is written before every destructive operation
- [ ] `RollbackEngine.rollback()` correctly restores files in LIFO order
- [ ] Integration test: rollback after a 3-task sequence restores all state

---

## Phase 6 — AgentScope Pipeline Wiring

**Goal**: Wire all agents into the official AgentScope multi-agent framework so conversations
and state are managed by the framework (not ad hoc).

### 6.1 — AgentScope Model Config

**File**: `configs/llm/agentscope_models.json`

```json
[
  {
    "model_type": "openai_chat",
    "config_name": "vllm_qwen",
    "model_name": "Qwen/Qwen3-4B-Instruct",
    "api_key": "EMPTY",
    "client_args": {
      "base_url": "http://localhost:8000/v1"
    },
    "generate_args": {
      "temperature": 0.1
    }
  }
]
```

### 6.2 — Agent Registration

**File**: `orion/orchestrator/pipeline.py`

```python
import agentscope
from agentscope.agents import UserAgent, DialogAgent
from agentscope.message import Msg
from agentscope.pipelines import SequentialPipeline, WhileLoopPipeline

agentscope.init(
    model_configs="configs/llm/agentscope_models.json",
    project="orion",
    run_id=task_id,
    save_log=True,
    save_dir="~/.orion/logs",
)

planner_agent = PlannerAgent(
    name="planner",
    model_config_name="vllm_qwen",
    sys_prompt=load_prompt("configs/agents/planner.yaml"),
)

human_agent = UserAgent(name="human")

deriver_agent = TaskDeriverAgent(name="deriver")

supervisor_agent = SupervisorAgent(
    name="supervisor",
    model_config_name="vllm_qwen",
)
```

### 6.3 — Message Protocol

All agents communicate via `agentscope.message.Msg`:

```python
# Planner → Human
plan_msg = Msg(
    name="planner",
    content=plan.model_dump_json(),
    role="assistant",
    metadata={"type": "IMPLEMENTATION_PLAN", "task_id": task_id},
)

# Human → Deriver
review_msg = Msg(
    name="human",
    content=review_result.model_dump_json(),
    role="user",
    metadata={"type": "HUMAN_REVIEW", "decision": "APPROVE"},
)
```

### 6.4 — Conversation History Preservation

- AgentScope maintains conversation history in `~/.orion/logs/{task_id}/`.
- On re-plan (REJECT loop), append previous plans and feedback to history so Qwen has full context.
- SupervisorAgent receives the full conversation history when making RETRY/ROLLBACK decisions.

**Checklist for Phase 6**:
- [ ] `agentscope.init()` called once at process start with correct model config
- [ ] All agents use `agentscope.message.Msg` for communication
- [ ] `UserAgent` blocks execution until human responds (not a timeout mock)
- [ ] Conversation history saved to `~/.orion/logs/{task_id}/`
- [ ] Re-plan loop passes full history (previous plans + feedback) to Qwen
- [ ] Integration test: full pipeline from prompt → HiTL → execution → verification

---

## Phase 7 — Observability & CLI Polish

### 7.1 — Structured Logging

**File**: `orion/observability/logger.py`

Every event emits a structured JSON log line:
```json
{
  "ts": "2025-04-16T10:00:00Z",
  "level": "INFO",
  "task_id": "abc-123",
  "subtask_id": "T-001",
  "event": "TOOL_CALL",
  "tool": "git_clone",
  "allowed": true,
  "duration_ms": 1234,
  "outcome": "SUCCESS"
}
```

### 7.2 — Tool Use Report

After each task completes, emit a tool use summary:
```
[T-001] Tool Usage Report:
  ✅ git_clone       — called 1x, succeeded
  ✅ fs_mkdir        — called 1x, succeeded
  🚫 write_file      — BLOCKED (not in allowed list)
  🚫 shell_exec      — BLOCKED (already used)
```

### 7.3 — CLI Commands Update

**New CLI commands**:
- `orion --plan "your prompt"` — Run planning only (no execution), display plan and exit.
- `orion --resume <task_id>` — Resume an interrupted task from last checkpoint.
- `orion --checklist <task_id>` — Display live checklist for a running task.
- `orion --rollback <task_id>` — Manually trigger rollback for a task.
- `orion --dry-run "your prompt"` — Full plan + HiTL but no actual tool execution.

**Checklist for Phase 7**:
- [ ] Every tool call emits a structured JSON log line
- [ ] Tool use report displayed after each task
- [ ] All new CLI flags work
- [ ] `--resume` correctly restores state from `tasks.json` + `checklist.json`
- [ ] `--dry-run` runs full pipeline with all tools mocked

---

## Data Contracts & Schemas

### ImplementationPlan
```python
class TaskCategory(str, Enum):
    SHELL = "SHELL"; FILE = "FILE"; GIT = "GIT"
    BROWSER = "BROWSER"; VISION = "VISION"; API = "API"; MEMORY = "MEMORY"

class ComplexityLevel(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"

class TaskNode(BaseModel):
    id: str                           # "T-001"
    title: str
    description: str
    category: TaskCategory
    allowed_tools: list[str]
    prerequisites: list[str]          # IDs of tasks that must be DONE first
    success_criteria: list[str]
    estimated_duration_sec: int
    is_destructive: bool = False

class ImplementationPlan(BaseModel):
    task_id: str
    original_prompt: str
    goal_summary: str
    estimated_complexity: ComplexityLevel
    tasks: list[TaskNode]
    rollback_strategy: str
```

### TaskStatus
```python
class TaskStatus(str, Enum):
    PENDING  = "PENDING"
    LOCKED   = "LOCKED"   # prerequisites not met
    RUNNING  = "RUNNING"
    DONE     = "DONE"
    FAILED   = "FAILED"
    SKIPPED  = "SKIPPED"
    ROLLED_BACK = "ROLLED_BACK"
```

### SupervisorDecision
```python
class SupervisorDecision(str, Enum):
    COMPLETE  = "COMPLETE"   # all done
    RETRY     = "RETRY"      # try same task again (max 3)
    ROLLBACK  = "ROLLBACK"   # undo and fail
    ESCALATE  = "ESCALATE"   # human needed
```

---

## File Diff Map

Files to **create** (new):

```
orion/
├── agents/
│   ├── planner_agent.py       [NEW] — Phase 1
│   ├── human_review_agent.py  [NEW] — Phase 1
│   ├── task_deriver.py        [NEW] — Phase 2
│   └── verifier_agent.py      [NEW] — Phase 5
├── executor/
│   ├── dispatcher.py          [NEW] — Phase 3
│   ├── base_specialist.py     [NEW] — Phase 3
│   ├── shell_executor.py      [NEW] — Phase 3
│   ├── file_executor.py       [NEW] — Phase 3
│   ├── git_executor.py        [NEW] — Phase 3
│   ├── browser_executor.py    [NEW] — Phase 3
│   ├── vision_executor.py     [NEW] — Phase 3
│   ├── api_executor.py        [NEW] — Phase 3
│   └── memory_executor.py     [NEW] — Phase 3
├── orchestrator/
│   ├── dag.py                 [NEW] — Phase 2
│   ├── checklist_gate.py      [NEW] — Phase 4
│   └── execution_loop.py      [NEW] — Phase 4
├── safety/
│   ├── tool_guard.py          [NEW] — Phase 3
│   └── rollback_engine.py     [NEW] — Phase 5 (extends existing)
├── cli/
│   └── checklist_display.py   [NEW] — Phase 4
└── core/
    └── schemas.py             [NEW] — All shared Pydantic models
```

Files to **modify** (existing):

```
orion/llm/vllm_provider.py     [MOD] — Add guided_json + structured output
orion/llm/router.py            [MOD] — Add provider logging + backoff
orion/tools/registry.py        [MOD] — Full manifest + remove bare write_file
orion/orchestrator/pipeline.py [MOD] — Wire AgentScope agents
orion_cli.py                   [MOD] — Add new CLI flags
configs/agents/planner.yaml    [MOD] — New structured system prompt
configs/llm/agentscope_models.json [NEW] — AgentScope model config
configs/safety/permissions.yaml    [MOD] — Per-category tool allowlists
```

---

## Risk Register

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Qwen 4B cannot reliably produce valid JSON plans | HIGH | HIGH | Use `guided_json` grammar + schema validation; fallback to Groq for planning calls |
| Human-in-the-loop adds latency users find frustrating | MEDIUM | MEDIUM | Add `--auto-approve` flag for CI/batch use; set 60s timeout with auto-approve |
| ToolGuard too strict — blocks legitimate retries | LOW | MEDIUM | Only block dedup on SUCCESS; failed tools can be retried |
| AgentScope `UserAgent` blocks async event loop | MEDIUM | HIGH | Run `UserAgent` in a separate thread; use `asyncio.run_in_executor` |
| Checklist deadlock if task fails and no rollback path | MEDIUM | HIGH | Deadlock detection raises error + auto-escalates to human |
| Qwen 4B produces tasks with unsupported categories | LOW | LOW | Schema validation rejects unknown categories; default to SHELL |

---

## Testing Strategy

### Unit Tests (`tests/unit/`)
- `test_tool_guard.py` — 20 scenarios covering allow/block/dedup
- `test_dag.py` — Topological sort, cycle detection, ready tasks
- `test_checklist_gate.py` — Prerequisites, locking, DONE propagation
- `test_verifier.py` — Pass/fail for each criterion type
- `test_planner_schema.py` — JSON output validation for 10 diverse prompts

### Integration Tests (`tests/integration/`)
- `test_full_pipeline.py` — Prompt → Plan → HiTL (mocked) → Execute → Verify
- `test_tool_guard_e2e.py` — Real dispatch with guard; verify no forbidden tools called
- `test_rollback.py` — 3-task sequence; fail task 2; verify task 1 state restored

### Evaluation Suite (`tests/eval/`)
- 20 diverse prompts spanning all 7 task categories
- Measure: plan quality score, tool diversity, task success rate, HiTL approval rate
- Baseline: current broken behavior (write_file for everything)
- Target: ≥80% task success rate, 0 forbidden tool calls

---

## Implementation Order (Sprint Map)

| Sprint | Phases | Duration | Deliverable |
|--------|--------|----------|-------------|
| S-1 | Phase 0 | 2 days | Qwen produces structured JSON reliably |
| S-2 | Phase 1 | 3 days | Plan → HiTL approve → re-plan cycle working |
| S-3 | Phase 2 | 2 days | tasks.json + checklist.json written atomically |
| S-4 | Phase 3 | 4 days | ToolGuard + all 7 specialists implemented |
| S-5 | Phase 4 | 2 days | Checklist-gated execution loop + live display |
| S-6 | Phase 5 | 2 days | Verifier + rollback working |
| S-7 | Phase 6 | 2 days | Full AgentScope wiring |
| S-8 | Phase 7 | 2 days | CLI polish + observability |
| **Total** | | **~19 days** | **Professional ORION harness** |
