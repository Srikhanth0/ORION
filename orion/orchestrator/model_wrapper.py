"""ORION model wrapper — bridges AdaptiveLLMRouter ↔ AgentScope.

Implements ``agentscope.model.ChatModelBase`` so that all AgentScope
agents transparently use ORION's tiered LLM backend (vLLM → Groq →
OpenRouter) via the AdaptiveLLMRouter.

Module Contract
---------------
- **Inputs**: AgentScope ``Msg`` objects (converted to OpenAI format).
- **Outputs**: ``ChatResponse`` compatible with AgentScope agents.

Depends On
----------
- ``agentscope.model`` (ChatModelBase, ChatResponse)
- ``agentscope.message`` (TextBlock)
- ``orion.llm.router`` (AdaptiveLLMRouter)
"""

from __future__ import annotations

from typing import Any

from agentscope.message import TextBlock
from agentscope.model import ChatModelBase, ChatResponse

from orion.llm.router import AdaptiveLLMRouter


class OrionModelWrapper(ChatModelBase):
    """AgentScope model wrapper that delegates to AdaptiveLLMRouter.

    Translates between AgentScope's Msg/ChatResponse format and
    ORION's OpenAI-style messages/LLMResponse format.

    Args:
        router: The shared AdaptiveLLMRouter instance.
        model_name: Display name for this model wrapper.
        default_model: Default model identifier override.
        temperature: Default temperature.
        max_tokens: Default max tokens.
    """

    def __init__(
        self,
        router: AdaptiveLLMRouter,
        model_name: str = "orion-adaptive",
        default_model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model_name=model_name, stream=False)
        self._router = router
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def __call__(
        self,
        messages: list[dict[str, Any]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse:
        """Route a chat completion through ORION's adaptive backend.

        Args:
            messages: OpenAI-format messages list.
            *args: Additional positional args (ignored).
            **kwargs: Overrides for model, temperature, max_tokens,
                response_format, preferred_provider.

        Returns:
            AgentScope ChatResponse with content as TextBlock.
        """
        if messages is None:
            messages = []

        # Extract ORION-specific kwargs
        model = kwargs.pop("model", self._default_model)
        temperature = kwargs.pop("temperature", self._temperature)
        max_tokens = kwargs.pop("max_tokens", self._max_tokens)
        response_format = kwargs.pop("response_format", None)
        preferred_provider = kwargs.pop("preferred_provider", None)

        # Call the adaptive router
        llm_response = await self._router.chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            preferred_provider=preferred_provider,
        )

        # Convert to AgentScope ChatResponse
        return ChatResponse(
            content=[TextBlock(text=llm_response.content)],
            metadata={
                "provider": llm_response.provider,
                "model": llm_response.model,
                "input_tokens": llm_response.input_tokens,
                "output_tokens": llm_response.output_tokens,
                "latency_ms": llm_response.latency_ms,
                "cost_usd": llm_response.cost_usd,
            },
        )

    def format_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pass through — messages are already in OpenAI format.

        Args:
            messages: Chat messages.

        Returns:
            Same messages, unchanged.
        """
        return messages
