"""Base agent for all ORION HiClaw agents.

Provides shared utilities for orion_meta extraction, Jinja2 prompt
rendering, structured JSON output parsing, and observability logging.

Module Contract
---------------
- Subclassed by PlannerAgent, ExecutorAgent, VerifierAgent, SupervisorAgent.
- Handles orion_meta propagation between agents.
- Renders prompts from Jinja2 templates.

Depends On
----------
- ``agentscope.agent`` (AgentBase)
- ``agentscope.message`` (Msg)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from agentscope.agent import AgentBase
from agentscope.message import Msg

logger = structlog.get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class BaseOrionAgent(AgentBase):
    """Base class for all ORION HiClaw agents.

    Provides shared utilities for:
    - ``orion_meta`` extraction and propagation
    - Jinja2 prompt template rendering
    - Structured JSON output parsing from LLM responses
    - Observability logging

    Args:
        agent_name: Human-readable name for this agent.
        model: AgentScope model wrapper (OrionModelWrapper).
        prompt_template: Name of the Jinja2 template file.
    """

    def __init__(
        self,
        agent_name: str,
        model: Any = None,
        prompt_template: str | None = None,
    ) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._model = model
        self._prompt_template = prompt_template
        self._template_cache: dict[str, Any] = {}

    def _get_orion_meta(self, msg: Msg) -> dict[str, Any]:
        """Extract orion_meta from a message, with defaults.

        Args:
            msg: AgentScope Msg object.

        Returns:
            orion_meta dict with all required fields.
        """
        metadata = getattr(msg, "metadata", None) or {}
        meta = metadata.get("orion_meta", {})
        return {
            "task_id": meta.get("task_id", "unknown"),
            "subtask_id": meta.get("subtask_id"),
            "step_index": meta.get("step_index", 0),
            "retry_count": meta.get("retry_count", 0),
            "rollback_available": meta.get("rollback_available", False),
            "trace_id": meta.get("trace_id", ""),
            "context": meta.get("context", {}),
        }

    def _set_orion_meta(
        self, msg: Msg, updates: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Copy and optionally update orion_meta for outgoing messages.

        Args:
            msg: Source message to copy meta from.
            updates: Fields to override in the copy.

        Returns:
            Updated orion_meta dict.
        """
        meta = self._get_orion_meta(msg)
        if updates:
            meta.update(updates)
        return meta

    def _render_prompt(
        self,
        template_name: str | None = None,
        **variables: Any,
    ) -> str:
        """Render a Jinja2 prompt template with variables.

        Args:
            template_name: Override template filename.
            **variables: Template variables.

        Returns:
            Rendered prompt string.
        """
        name = template_name or self._prompt_template
        if name is None:
            return ""

        if name not in self._template_cache:
            import jinja2

            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(_PROMPTS_DIR)),
                autoescape=False,
            )
            self._template_cache[name] = env.get_template(name)

        return self._template_cache[name].render(**variables)

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        """Call the LLM through the model wrapper.

        Args:
            messages: OpenAI-format messages.
            **kwargs: Additional model kwargs.

        Returns:
            LLM response content string.
        """
        if self._model is None:
            msg = "No model configured for agent"
            raise RuntimeError(msg)

        response = await self._model(messages=messages, **kwargs)
        # Extract text from ChatResponse blocks
        if hasattr(response, "content") and response.content:
            parts = []
            for block in response.content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return ""

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling code fences.

        Args:
            text: Raw LLM output that should contain JSON.

        Returns:
            Parsed dict.

        Raises:
            ValueError: If JSON parsing fails.
        """
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (fences)
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning(
                "json_parse_failed",
                agent=self.agent_name,
                error=str(exc),
                text_preview=cleaned[:200],
            )
            raise ValueError(
                f"Failed to parse JSON from LLM output: {exc}"
            ) from exc

    def _make_reply(
        self,
        content: str,
        source_msg: Msg,
        meta_updates: dict[str, Any] | None = None,
    ) -> Msg:
        """Create a reply Msg with orion_meta propagated.

        Args:
            content: Response content string.
            source_msg: Original message to propagate meta from.
            meta_updates: Optional meta field overrides.

        Returns:
            New Msg with propagated orion_meta.
        """
        meta = self._set_orion_meta(source_msg, meta_updates)
        return Msg(
            name=self.agent_name,
            role="assistant",
            content=content,
            metadata={"orion_meta": meta},
        )
