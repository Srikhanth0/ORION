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

from pathlib import Path
from typing import Any

import structlog
from agentscope.agent import AgentBase
from agentscope.message import Msg

from orion.core.utils.json_utils import parse_json

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
        self._prompt_template = prompt_template
        self._template_cache: dict[str, Any] = {}

        # ORION-FIX: Auto-provision model via build_model() when None
        self._model = model
        if self._model is None:
            try:
                from orion.agentscope_config import build_model

                self._model = build_model()
            except Exception as exc:
                logger.warning(
                    "model_auto_provision_failed",
                    agent=agent_name,
                    error=str(exc),
                )

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

    def _set_orion_meta(self, msg: Msg, updates: dict[str, Any] | None = None) -> dict[str, Any]:
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

        return self._template_cache[name].render(**variables)  # type: ignore[no-any-return]

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
        # AgentScope OpenAIChatModel has a bug in v1.0.18 where stream=False throws an async context manager error.  # noqa: E501
        # So we omit stream=False and consume the returned async_generator.
        if "stream" in kwargs:
            del kwargs["stream"]

        response = await self._model(messages=messages, **kwargs)

        # If response is an async generator (e.g. streaming fallback), consume it
        import inspect

        if inspect.isasyncgen(response):
            last_content = ""
            async for chunk in response:
                content = chunk.get("content")
                if isinstance(content, str):
                    last_content = content
                elif isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            parts.append(block["text"])
                        elif hasattr(block, "text"):
                            parts.append(block.text)
                    last_content = "".join(parts)
            return last_content

        if getattr(response, "text", None):
            return response.text  # type: ignore[no-any-return]

        # Fallback if content handles are used
        content = getattr(response, "content", None)
        if isinstance(content, str):
            return content
        elif content:
            parts = []
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
                elif hasattr(block, "text"):
                    parts.append(block.text)
            return "\n".join(parts)
        return ""

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling code fences."""
        return parse_json(text)

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
