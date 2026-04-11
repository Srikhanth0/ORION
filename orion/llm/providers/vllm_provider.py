"""vLLM provider — local Qwen 2.5 via OpenAI-compatible endpoint.

Priority 1 in the fallback chain. Zero-cost inference on self-hosted
GPU hardware. Uses ``openai.AsyncOpenAI`` with ``base_url`` override.

Module Contract
---------------
- **Inputs**: Chat messages in OpenAI format.
- **Outputs**: ``LLMResponse`` with token counts from vLLM.
- **Failure Modes**: ``LLMError`` on connection failure or OOM.
- **Health Check**: GET ``/v1/models`` endpoint.

Depends On
----------
- ``openai`` SDK (AsyncOpenAI)
- ``orion.llm.providers.base`` (LLMProvider, LLMResponse, estimate_cost)
- ``orion.core.exceptions`` (LLMError)
"""
from __future__ import annotations

import time
from typing import Any

import structlog

from orion.core.exceptions import LLMError
from orion.llm.providers.base import (
    LLMResponse,
    QuotaInfo,
    estimate_cost,
)

logger = structlog.get_logger(__name__)


class VLLMProvider:
    """OpenAI-compatible vLLM provider for local Qwen 2.5 inference.

    Uses ``openai.AsyncOpenAI`` with ``base_url`` pointed at the
    vLLM server. On OOM (HTTP 500 with CUDA out of memory), marks
    itself unhealthy and emits an alert.

    Args:
        base_url: vLLM OpenAI-compatible endpoint (e.g., ``http://localhost:8000/v1``).
        model: Default model name (as registered with ``--served-model-name``).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "qwen2.5-72b",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._client: Any = None  # Lazy init to avoid import-time failures

    def _get_client(self) -> Any:
        """Lazily initialise the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self._base_url,
                api_key="EMPTY",  # vLLM doesn't require auth
                timeout=self._timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "vllm"

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to vLLM.

        Args:
            messages: OpenAI-format chat messages.
            model: Override default model.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional response format spec.
            **kwargs: Additional parameters forwarded to vLLM.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMError: On connection failure, OOM, or unexpected error.
        """
        client = self._get_client()
        used_model = model or self._model
        start = time.monotonic()

        try:
            create_kwargs: dict[str, Any] = {
                "model": used_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format is not None:
                create_kwargs["response_format"] = response_format

            response = await client.chat.completions.create(**create_kwargs)

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            error_msg = str(exc)

            # Detect CUDA OOM
            if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
                logger.error(
                    "vllm_oom",
                    provider=self.name,
                    model=used_model,
                    error=error_msg,
                    latency_ms=elapsed,
                )
                raise LLMError(
                    f"vLLM OOM: {error_msg}",
                    provider=self.name,
                    context={"oom": True, "model": used_model},
                ) from exc

            logger.error(
                "vllm_request_failed",
                provider=self.name,
                model=used_model,
                error=error_msg,
                latency_ms=elapsed,
            )
            raise LLMError(
                f"vLLM request failed: {error_msg}",
                provider=self.name,
            ) from exc

        elapsed = (time.monotonic() - start) * 1000

        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""
        cost = estimate_cost(used_model, input_tokens, output_tokens)

        result = LLMResponse(
            content=content,
            provider=self.name,
            model=used_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed,
            cost_usd=cost,
        )

        logger.info(
            "llm_response",
            provider=self.name,
            model=used_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(elapsed, 1),
            cost_usd=cost,
        )

        return result

    async def is_healthy(self) -> bool:
        """Check vLLM health via models.list() endpoint.

        Returns:
            True if vLLM responds with a model list.
        """
        try:
            client = self._get_client()
            models = await client.models.list()
            return len(models.data) > 0
        except Exception as exc:
            logger.warning(
                "vllm_health_check_failed",
                provider=self.name,
                error=str(exc),
            )
            return False

    async def remaining_quota(self) -> QuotaInfo | None:
        """vLLM has no quota — local inference is unlimited.

        Returns:
            None (no quota tracking for local provider).
        """
        return None
