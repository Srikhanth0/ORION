"""Groq provider — free-tier cloud LLM with rate-limit tracking.

Priority 2 in the fallback chain. Parses ``x-ratelimit-*`` headers
from every response and caches them in the shared QuotaTracker.

Module Contract
---------------
- **Inputs**: Chat messages in OpenAI format.
- **Outputs**: ``LLMResponse`` with quota headers parsed.
- **Failure Modes**: ``LLMError`` on API failure, ``QuotaExceededError``
  on 429 response.

Depends On
----------
- ``groq`` SDK (AsyncGroq)
- ``orion.llm.providers.base`` (LLMProvider, LLMResponse, estimate_cost)
- ``orion.llm.quota`` (QuotaTracker)
- ``orion.core.exceptions`` (LLMError, QuotaExceededError)
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

import structlog

from orion.core.exceptions import LLMError, QuotaExceededError
from orion.llm.providers.base import (
    LLMResponse,
    QuotaInfo,
    estimate_cost,
)
from orion.llm.quota import QuotaTracker

logger = structlog.get_logger(__name__)


class GroqProvider:
    """Groq cloud provider with quota tracking.

    Uses ``groq.AsyncGroq`` SDK. After every response, parses
    rate-limit headers and updates the shared QuotaTracker.

    Args:
        api_key: Groq API key (``gsk_...``).
        model: Default model for text requests.
        vision_model: Default model for vision requests.
        quota_tracker: Shared QuotaTracker instance for rate-limit caching.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        quota_tracker: QuotaTracker | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._vision_model = vision_model
        self._quota_tracker = quota_tracker or QuotaTracker("groq")
        self._timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialise the Groq async client."""
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "groq"

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
        """Send a chat completion request to Groq.

        Parses rate-limit headers from the response and updates
        the QuotaTracker.

        Args:
            messages: OpenAI-format chat messages.
            model: Override default model.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional response format spec.
            **kwargs: Additional parameters.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMError: On API failure.
            QuotaExceededError: On 429 rate-limit response.
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

            # Detect rate-limit (429)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                # Try to extract retry-after
                retry_after: float | None = None
                if hasattr(exc, "response") and exc.response is not None:
                    retry_header = exc.response.headers.get("retry-after")
                    if retry_header:
                        with contextlib.suppress(ValueError, TypeError):
                            retry_after = float(retry_header)

                logger.warning(
                    "groq_rate_limited",
                    provider=self.name,
                    model=used_model,
                    retry_after=retry_after,
                )
                raise QuotaExceededError(
                    f"Groq rate limit exceeded: {error_msg}",
                    provider=self.name,
                    retry_after_seconds=retry_after,
                ) from exc

            logger.error(
                "groq_request_failed",
                provider=self.name,
                model=used_model,
                error=error_msg,
                latency_ms=elapsed,
            )
            raise LLMError(
                f"Groq request failed: {error_msg}",
                provider=self.name,
            ) from exc

        elapsed = (time.monotonic() - start) * 1000

        # Parse rate-limit headers from the raw response
        raw_headers: dict[str, str] = {}
        if hasattr(response, "_raw_response") and response._raw_response is not None:
            for key, value in response._raw_response.headers.items():
                if key.lower().startswith("x-ratelimit"):
                    raw_headers[key.lower()] = value

        # Update quota tracker
        if raw_headers:
            await self._quota_tracker.update_from_headers(raw_headers)

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
            raw_headers=raw_headers,
        )

        logger.info(
            "llm_response",
            provider=self.name,
            model=used_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(elapsed, 1),
            cost_usd=cost,
            remaining_requests=raw_headers.get("x-ratelimit-remaining-requests"),
        )

        return result

    async def is_healthy(self) -> bool:
        """Check Groq health via models.list() API.

        Returns:
            True if Groq API responds successfully.
        """
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception as exc:
            logger.warning(
                "groq_health_check_failed",
                provider=self.name,
                error=str(exc),
            )
            return False

    async def remaining_quota(self) -> QuotaInfo | None:
        """Return cached quota info from the QuotaTracker.

        Returns:
            QuotaInfo snapshot from the last API response headers.
        """
        return await self._quota_tracker.get_quota()
