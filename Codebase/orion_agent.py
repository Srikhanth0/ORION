"""
ORION v1.0 — Core Working Agent Model
======================================
Drop this file at the project root and run: python orion_agent.py "your instruction"

Requires:
  pip install agentscope mcp chromadb httpx pyautogui pillow rich prompt_toolkit groq

Set environment variables:
  GROQ_API_KEY      — required for LLM (or use OPENROUTER_API_KEY)
  VISION_API_URL    — optional, ngrok URL from Colab vision server
  GITHUB_PAT        — optional, for GitHub tool calls
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# 1. RESULT TYPE
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Result:
    success: bool
    data: Any = None
    error: str = ""

    @classmethod
    def ok(cls, data: Any = None) -> Result:
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> Result:
        return cls(success=False, error=error)

    def __repr__(self):
        if self.success:
            preview = str(self.data)[:120] + ("…" if len(str(self.data)) > 120 else "")
            return f"Result.ok({preview})"
        return f"Result.fail({self.error})"


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM CLIENT (Groq-based, with OpenRouter fallback)
# ─────────────────────────────────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, threshold: int = 3, open_duration: float = 60.0):
        self.threshold = threshold
        self.open_duration = open_duration
        self.failures = 0
        self.state = CircuitState.CLOSED
        self._opened_at: float = 0.0

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            print(f"[CircuitBreaker] OPEN after {self.failures} failures")

    def record_success(self):
        self.failures = 0
        self.state = CircuitState.CLOSED

    def is_available(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.time() - self._opened_at > self.open_duration:
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN: allow one probe


class OrionLLM:
    """
    Adaptive LLM router: vLLM → Groq → OpenRouter (with circuit breakers).
    Falls back gracefully so the agent always has a provider.
    """

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {
            "groq": CircuitBreaker(),
            "openrouter": CircuitBreaker(),
        }
        self._groq_key = os.getenv("GROQ_API_KEY", "")
        self._or_key = os.getenv("OPENROUTER_API_KEY", "")

    async def complete(self, messages: list[dict], json_mode: bool = False) -> str:
        """Call the best available provider and return the assistant text."""
        # Try Groq first
        if self._groq_key and self._breakers["groq"].is_available():
            result = await self._call_groq(messages, json_mode)
            if result.success:
                self._breakers["groq"].record_success()
                return result.data
            self._breakers["groq"].record_failure()
            print(f"[LLM] Groq failed: {result.error} — trying OpenRouter")

        # Try OpenRouter
        if self._or_key and self._breakers["openrouter"].is_available():
            result = await self._call_openrouter(messages, json_mode)
            if result.success:
                self._breakers["openrouter"].record_success()
                return result.data
            self._breakers["openrouter"].record_failure()

        raise RuntimeError(
            "All LLM providers unavailable. Check GROQ_API_KEY or OPENROUTER_API_KEY."
        )

    async def _call_groq(self, messages: list[dict], json_mode: bool) -> Result:
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self._groq_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "llama-3.1-70b-versatile",
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.2,
            }
            if json_mode:
                body["response_format"] = {"type": "json_object"}

            async with httpx.AsyncClient(timeout=45) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                return Result.ok(data["choices"][0]["message"]["content"])
        except Exception as e:
            return Result.fail(str(e))

    async def _call_openrouter(self, messages: list[dict], json_mode: bool) -> Result:
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self._or_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": "anthropic/claude-3-haiku",
                "messages": messages,
                "max_tokens": 2048,
            }
            async with httpx.AsyncClient(timeout=45) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                return Result.ok(data["choices"][0]["message"]["content"])
        except Exception as e:
            return Result.fail(str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MEMORY LAYER
# ─────────────────────────────────────────────────────────────────────────────


class WorkingMemory:
    """Sliding-window in-process context for the current session."""

    def __init__(self, max_messages: int = 20):
        self._messages: deque[dict] = deque(maxlen=max_messages)

    def add(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_context(self) -> list[dict]:
        return list(self._messages)

    def clear(self):
        self._messages.clear()


class LongTermMemory:
    """
    ChromaDB-backed persistent semantic memory.
    Falls back to a no-op in-memory store if chromadb is not installed.
    """

    def __init__(self, persist_path: str = ".orion_memory"):
        self._available = False
        try:
            import chromadb

            self._client = chromadb.PersistentClient(path=persist_path)
            self._collection = self._client.get_or_create_collection("orion_tasks")
            self._available = True
        except ImportError:
            print("[Memory] chromadb not installed — long-term memory disabled")
            self._store: list[dict] = []

    def store(self, task_id: str, summary: str, metadata: dict | None = None):
        if not self._available:
            self._store.append({"id": task_id, "summary": summary})
            return
        self._collection.upsert(
            ids=[task_id],
            documents=[summary],
            metadatas=[metadata or {}],
        )

    def retrieve(self, query: str, n: int = 5) -> list[dict]:
        if not self._available:
            return self._store[-n:]
        try:
            results = self._collection.query(query_texts=[query], n_results=n)
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=False)
            ]
        except Exception:
            return []


# ─────────────────────────────────────────────────────────────────────────────
# 4. SAFETY GATE
# ─────────────────────────────────────────────────────────────────────────────

BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":(){ :|:& };:",  # fork bomb
    r"dd\s+if=.*of=/dev",  # disk wipe
    r"chmod\s+777\s+/",
]

HIGH_RISK_PATTERNS = [
    r"rm\s+-",
    r"sudo\s+",
    r"git\s+push\s+.*--force",
    r"DROP\s+TABLE",
    r"format\s+[A-Z]:",
]

ALWAYS_APPROVE_VISION = ["click_element", "type_text", "press_key"]


class SafetyGate:
    def __init__(self, strict_mode: bool = True):
        self.strict = strict_mode

    def check(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        In strict mode, high-risk ops return (False, 'requires_approval').
        In auto mode, only blocked patterns are denied.
        """
        cmd_text = json.dumps(arguments)

        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, cmd_text, re.IGNORECASE):
                return False, f"Blocked pattern detected: {pattern}"

        if tool_name in ALWAYS_APPROVE_VISION and self.strict:
            return False, "requires_human_approval"

        for pattern in HIGH_RISK_PATTERNS:
            if re.search(pattern, cmd_text, re.IGNORECASE) and self.strict:
                return False, "requires_human_approval"

        return True, "ok"

    def request_approval(self, tool_name: str, arguments: dict) -> bool:
        print("\n[SAFETY] High-risk action requires approval:")
        print(f"  Tool: {tool_name}")
        print(f"  Args: {json.dumps(arguments, indent=2)}")
        response = input("  Approve? [y/N]: ").strip().lower()
        return response == "y"


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL REGISTRY (OS + Vision — extensible to full MCP)
# ─────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Built-in local tool implementations for fast prototyping.
    In v1.0 production: replace each tool fn with MCPStdioClient calls.
    """

    def __init__(self, vision_url: str = ""):
        self._vision_url = vision_url or os.getenv("VISION_API_URL", "")
        self._tools = self._register()

    def _register(self) -> dict[str, dict]:
        return {
            # ── OS Tools ──────────────────────────────────────────────────
            "list_files": {
                "category": "os_tools",
                "description": "List files in a directory. Args: path (str), pattern (str, optional)",
                "fn": self._list_files,
            },
            "read_file": {
                "category": "os_tools",
                "description": "Read file content. Args: path (str)",
                "fn": self._read_file,
            },
            "write_file": {
                "category": "os_tools",
                "description": "Write content to a file. Args: path (str), content (str)",
                "fn": self._write_file,
            },
            "execute_shell": {
                "category": "os_tools",
                "description": "Run a shell command. Args: command (str)",
                "fn": self._execute_shell,
            },
            "find_pattern": {
                "category": "os_tools",
                "description": "Find lines matching a pattern. Args: path (str), pattern (str)",
                "fn": self._find_pattern,
            },
            # ── Vision Tools ──────────────────────────────────────────────
            "take_screenshot": {
                "category": "vision_tools",
                "description": "Take a screenshot. Returns base64 PNG. Args: none",
                "fn": self._take_screenshot,
            },
            "analyze_screen": {
                "category": "vision_tools",
                "description": "Analyze current screen with Qwen2.5-VL. Args: prompt (str, optional)",
                "fn": self._analyze_screen,
            },
            "click_element": {
                "category": "vision_tools",
                "description": "Click a UI element by description. Args: description (str)",
                "fn": self._click_element,
            },
            "type_text": {
                "category": "vision_tools",
                "description": "Type text at current cursor. Args: text (str)",
                "fn": self._type_text,
            },
            "press_key": {
                "category": "vision_tools",
                "description": "Press keyboard key. Args: key (str, e.g. 'enter', 'ctrl+c')",
                "fn": self._press_key,
            },
        }

    def list_tools(self) -> list[dict]:
        return [
            {"name": k, "category": v["category"], "description": v["description"]}
            for k, v in self._tools.items()
        ]

    async def call(self, tool_name: str, arguments: dict) -> Result:
        if tool_name not in self._tools:
            return Result.fail(f"Unknown tool: {tool_name}")
        try:
            fn = self._tools[tool_name]["fn"]
            result = await fn(**arguments)
            return Result.ok(result)
        except Exception as e:
            return Result.fail(f"{tool_name} error: {e}")

    # ── OS Tool Implementations ────────────────────────────────────────────

    async def _list_files(self, path: str = ".", pattern: str = "*") -> list[str]:
        p = Path(path).expanduser()
        return [str(f) for f in p.glob(f"**/{pattern}") if f.is_file()][:200]

    async def _read_file(self, path: str) -> str:
        return Path(path).expanduser().read_text(errors="replace")

    async def _write_file(self, path: str, content: str) -> str:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Written {len(content)} chars to {path}"

    async def _execute_shell(self, command: str) -> str:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        output = stdout.decode(errors="replace")
        if stderr:
            output += "\n[stderr]: " + stderr.decode(errors="replace")
        return output

    async def _find_pattern(self, path: str, pattern: str) -> list[str]:
        content = Path(path).expanduser().read_text(errors="replace")
        matches = []
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(pattern, line, re.IGNORECASE):
                matches.append(f"L{i}: {line.strip()}")
        return matches

    # ── Vision Tool Implementations ─────────────────────────────────────────

    async def _take_screenshot(self) -> str:
        try:
            from PIL import ImageGrab

            img = ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()
        except ImportError:
            return "error: pillow not installed"
        except Exception as e:
            return f"error: {e}"

    async def _analyze_screen(self, prompt: str = "") -> dict:
        if not self._vision_url:
            return {
                "error": "VISION_API_URL not set",
                "hint": "Start the Colab vision server and set VISION_API_URL in .env",
            }
        import httpx

        img_b64 = await self._take_screenshot()
        if img_b64.startswith("error"):
            return {"error": img_b64}
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                f"{self._vision_url}/analyze",
                json={
                    "image_base64": img_b64,
                    "prompt": prompt
                    or (
                        "Describe this screen. List all visible UI elements "
                        "with their approximate pixel coordinates."
                    ),
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def _click_element(self, description: str) -> dict:
        try:
            import pyautogui
        except ImportError:
            return {"success": False, "error": "pyautogui not installed"}

        result = await self._analyze_screen(
            prompt=(
                f"Find the UI element matching: '{description}'. "
                'Return ONLY valid JSON: {"x": <int>, "y": <int>, "found": true|false}'
            )
        )
        text = result.get("result", "")
        match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if not match:
            return {"success": False, "error": "Model did not return coordinates"}
        try:
            coords = json.loads(match.group())
        except json.JSONDecodeError:
            return {"success": False, "error": "Could not parse coordinate JSON"}
        if not coords.get("found"):
            return {"success": False, "error": f"Element '{description}' not found on screen"}
        pyautogui.click(int(coords["x"]), int(coords["y"]))
        return {"success": True, "clicked_at": {"x": coords["x"], "y": coords["y"]}}

    async def _type_text(self, text: str) -> dict:
        try:
            import pyautogui

            pyautogui.write(text, interval=0.04)
            return {"success": True, "typed": text}
        except ImportError:
            return {"success": False, "error": "pyautogui not installed"}

    async def _press_key(self, key: str) -> dict:
        try:
            import pyautogui

            pyautogui.hotkey(*key.split("+"))
            return {"success": True, "key": key}
        except ImportError:
            return {"success": False, "error": "pyautogui not installed"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. HICLAW AGENTS
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """You are the ORION Planner agent. Your job is to decompose a user instruction
into a DAG of concrete subtasks that can be executed by OS and vision tools.

Output ONLY valid JSON matching this schema (no markdown, no explanation):
{
  "subtasks": [
    {
      "id": "t1",
      "action": "human-readable description",
      "tool_name": "one of: list_files, read_file, write_file, execute_shell, find_pattern, take_screenshot, analyze_screen, click_element, type_text, press_key",
      "arguments": {},
      "depends_on": []
    }
  ]
}

Rules:
- Use concrete tool names from the list above only.
- Set depends_on to [] for tasks that can run immediately.
- Set depends_on to ["t1", "t2"] for tasks that need prior results.
- Minimize sequential dependencies: tasks without dependencies run in parallel.
- Keep subtasks atomic — one tool call each.
- Maximum 12 subtasks per plan.
"""

VERIFIER_SYSTEM = """You are the ORION Verifier agent. You receive the original instruction and the
results of all executed subtasks. Determine if the task was completed successfully.

Output ONLY valid JSON:
{"passed": true|false, "reason": "brief explanation"}
"""

SUPERVISOR_SYSTEM = """You are the ORION Supervisor agent. You receive the verifier's assessment.
Decide the next action.

Output ONLY valid JSON:
{"decision": "complete"|"retry"|"partial_complete", "note": "brief explanation"}

- complete: task fully accomplished
- retry: a critical subtask failed, replanning needed
- partial_complete: some subtasks succeeded but the goal is not fully met
"""


class PlannerAgent:
    def __init__(self, llm: OrionLLM, working_mem: WorkingMemory, long_mem: LongTermMemory):
        self.llm = llm
        self.wm = working_mem
        self.lm = long_mem

    async def plan(self, instruction: str, failure_context: str = "") -> list[dict]:
        memories = self.lm.retrieve(instruction)
        memory_text = ""
        if memories:
            memory_text = "\nRelevant past patterns:\n" + "\n".join(
                f"- {m.get('document', '')}" for m in memories[:3]
            )

        user_content = f"Instruction: {instruction}"
        if failure_context:
            user_content += f"\n\nPrevious attempt failed: {failure_context}"
        if memory_text:
            user_content += memory_text

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            *self.wm.get_context()[-6:],
            {"role": "user", "content": user_content},
        ]

        raw = await self.llm.complete(messages, json_mode=True)
        try:
            plan = json.loads(raw)
            subtasks = plan.get("subtasks", [])
            print(f"[Planner] Generated {len(subtasks)} subtasks")
            return subtasks
        except json.JSONDecodeError as e:
            print(f"[Planner] JSON parse error: {e}\nRaw: {raw[:200]}")
            return []


class ExecutorAgent:
    def __init__(self, registry: ToolRegistry, safety: SafetyGate):
        self.registry = registry
        self.safety = safety

    async def execute_subtask(self, subtask: dict, prior_results: dict) -> dict:
        tool_name = subtask.get("tool_name", "")
        arguments = subtask.get("arguments", {})

        # Inject prior results if referenced
        for k, v in arguments.items():
            if isinstance(v, str) and v.startswith("$result."):
                ref_id = v.split(".", 1)[1]
                if ref_id in prior_results:
                    arguments[k] = prior_results[ref_id].get("output", v)

        # Safety check
        allowed, reason = self.safety.check(tool_name, arguments)
        if not allowed:
            if reason == "requires_human_approval":
                approved = self.safety.request_approval(tool_name, arguments)
                if not approved:
                    return {
                        "id": subtask["id"],
                        "success": False,
                        "output": None,
                        "error": "Rejected by user",
                    }
            else:
                return {
                    "id": subtask["id"],
                    "success": False,
                    "output": None,
                    "error": f"Blocked: {reason}",
                }

        print(f"[Executor] Running {tool_name}({arguments})")
        result = await self.registry.call(tool_name, arguments)
        return {
            "id": subtask["id"],
            "success": result.success,
            "output": result.data,
            "error": result.error,
        }

    async def execute_dag(self, subtasks: list[dict]) -> dict[str, dict]:
        """Topological parallel execution using asyncio.gather."""
        results: dict[str, dict] = {}
        pending = {t["id"]: t for t in subtasks}
        max_iterations = len(subtasks) + 1  # Guard against cycles
        iterations = 0

        while pending and iterations < max_iterations:
            iterations += 1
            ready = [
                t
                for t in pending.values()
                if all(dep in results for dep in t.get("depends_on", []))
            ]
            if not ready:
                print("[Executor] Warning: unresolvable dependencies, forcing remaining tasks")
                ready = list(pending.values())[:1]

            coros = [self.execute_subtask(t, results) for t in ready]
            batch = await asyncio.gather(*coros, return_exceptions=True)

            for task, result in zip(ready, batch, strict=False):
                if isinstance(result, Exception):
                    results[task["id"]] = {
                        "id": task["id"],
                        "success": False,
                        "output": None,
                        "error": str(result),
                    }
                else:
                    results[task["id"]] = result
                del pending[task["id"]]
                status = "✓" if results[task["id"]]["success"] else "✗"
                print(f"  [{status}] {task['id']}: {task.get('action', task['tool_name'])}")

        return results


class VerifierAgent:
    def __init__(self, llm: OrionLLM):
        self.llm = llm

    async def verify(self, instruction: str, results: dict[str, dict]) -> dict:
        summary = json.dumps(
            {
                k: {
                    "success": v["success"],
                    "error": v.get("error"),
                    "output_preview": str(v.get("output", ""))[:200],
                }
                for k, v in results.items()
            },
            indent=2,
        )

        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": f"Instruction: {instruction}\n\nResults:\n{summary}"},
        ]
        raw = await self.llm.complete(messages, json_mode=True)
        try:
            verdict = json.loads(raw)
            passed = verdict.get("passed", False)
            reason = verdict.get("reason", "")
            status = "PASS" if passed else "FAIL"
            print(f"[Verifier] {status}: {reason}")
            return verdict
        except json.JSONDecodeError:
            return {"passed": False, "reason": f"Verifier JSON parse error: {raw[:100]}"}


class SupervisorAgent:
    def __init__(self, llm: OrionLLM):
        self.llm = llm

    async def decide(self, verdict: dict, attempt: int, max_attempts: int) -> dict:
        if attempt >= max_attempts:
            return {"decision": "partial_complete", "note": "Max retries reached"}

        messages = [
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Verifier result: {json.dumps(verdict)}\n"
                    f"Attempt {attempt} of {max_attempts}."
                ),
            },
        ]
        raw = await self.llm.complete(messages, json_mode=True)
        try:
            decision = json.loads(raw)
            print(f"[Supervisor] Decision: {decision.get('decision')} — {decision.get('note')}")
            return decision
        except json.JSONDecodeError:
            return {"decision": "partial_complete", "note": "Supervisor parse error"}


# ─────────────────────────────────────────────────────────────────────────────
# 7. PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────


class ORIONPipeline:
    """
    Full HiClaw pipeline: Planner → Executor (DAG) → Verifier → Supervisor.
    Supervisor can loop back to Planner on retry.
    """

    MAX_RETRIES = 3

    def __init__(self):
        self.llm = OrionLLM()
        self.working_mem = WorkingMemory(max_messages=20)
        self.long_mem = LongTermMemory()
        self.registry = ToolRegistry()
        self.safety = SafetyGate(strict_mode=True)

        self.planner = PlannerAgent(self.llm, self.working_mem, self.long_mem)
        self.executor = ExecutorAgent(self.registry, self.safety)
        self.verifier = VerifierAgent(self.llm)
        self.supervisor = SupervisorAgent(self.llm)

    async def run(self, instruction: str) -> dict:
        print(f"\n{'━'*60}")
        print(f"ORION v1.0 | Task: {instruction}")
        print(f"{'━'*60}\n")

        self.working_mem.add("user", instruction)
        failure_context = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            print(f"[Pipeline] Attempt {attempt}/{self.MAX_RETRIES}")

            # Plan
            subtasks = await self.planner.plan(instruction, failure_context)
            if not subtasks:
                return {"status": "failed", "error": "Planner produced no subtasks"}

            # Execute DAG
            results = await self.executor.execute_dag(subtasks)

            # Verify
            verdict = await self.verifier.verify(instruction, results)

            # Supervise
            decision = await self.supervisor.decide(verdict, attempt, self.MAX_RETRIES)

            if decision["decision"] == "complete":
                self.working_mem.add("assistant", f"Task completed: {decision.get('note')}")
                self.long_mem.store(
                    task_id=f"task_{int(time.time())}",
                    summary=instruction,
                    metadata={"status": "complete", "subtasks": len(subtasks)},
                )
                return {
                    "status": "complete",
                    "subtasks_run": len(subtasks),
                    "results": results,
                    "verdict": verdict,
                }

            elif decision["decision"] == "retry":
                failure_context = verdict.get("reason", "unknown failure")
                print(f"[Pipeline] Retrying — {failure_context}\n")
                continue

            else:  # partial_complete
                return {
                    "status": "partial_complete",
                    "subtasks_run": len(subtasks),
                    "results": results,
                    "verdict": verdict,
                    "note": decision.get("note"),
                }

        return {
            "status": "failed",
            "error": f"Task did not complete after {self.MAX_RETRIES} attempts",
        }

    def list_tools(self) -> list[dict]:
        return self.registry.list_tools()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║          ◈  ORION  v1.0  ◈  OS Automation Agent     ║
║     HiClaw · AgentScope · MCP · Qwen2.5-VL          ║
╚══════════════════════════════════════════════════════╝
  Commands: /tools  /quit  /help
  Or just type your instruction and press Enter.
"""


async def interactive_repl():
    """Rich + prompt_toolkit interactive REPL."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        session = PromptSession(history=InMemoryHistory())
        get_input = lambda: session.prompt_async("[ORION] › ")
    except ImportError:

        async def get_input():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: input("[ORION] › "))

    print(BANNER)
    pipeline = ORIONPipeline()

    while True:
        try:
            user_input = (await get_input()).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Goodbye.")
            break
        if user_input == "/help":
            print("  /tools  — list available tools")
            print("  /quit   — exit")
            print("  Or type any natural language instruction.")
            continue
        if user_input == "/tools":
            tools = pipeline.list_tools()
            for t in tools:
                print(f"  [{t['category']}] {t['name']}: {t['description']}")
            continue

        result = await pipeline.run(user_input)

        print(f"\n[Result] Status: {result['status']}")
        if result.get("results"):
            for task_id, r in result["results"].items():
                status_icon = "✓" if r["success"] else "✗"
                output_preview = str(r.get("output", ""))[:300]
                if len(output_preview) >= 300:
                    output_preview += "…"
                print(f"  {status_icon} {task_id}: {output_preview}")
        if result.get("error"):
            print(f"  Error: {result['error']}")
        print()


def main():
    """Entry point: single instruction from CLI args, or interactive REPL."""
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
        pipeline = ORIONPipeline()
        result = asyncio.run(pipeline.run(instruction))
        print(json.dumps(result, indent=2, default=str))
    else:
        asyncio.run(interactive_repl())


if __name__ == "__main__":
    main()
