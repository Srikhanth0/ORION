"""OpenRouter provider — pay-per-token cloud LLM fallback.

Priority 3 (last resort) in the fallback chain. Uses ``httpx.AsyncClient``
with required ``HTTP-Referer`` and ``X-Title`` headers. Budget-capped
via the shared QuotaTracker.

Module Contract
---------------
- **Inputs**: Chat messages in OpenAI format.
- **Outputs**: ``LLMResponse`` with cost tracking.
- **Failure Modes**: ``LLMError`` on API failure, ``QuotaExceededError``
  on 429 or budget exhaustion.

Depends On
----------
- ``httpx`` (async HTTP client)
- ``orion.llm.providers.base`` (LLMProvider, LLMResponse, estimate_cost)
- ``orion.llm.quota`` (QuotaTracker)
- ``orion.core.exceptions`` (LLMError, QuotaExceededError)
"""
from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from orion.core.exceptions import LLMError, QuotaExceededError
from orion.llm.providers.base import (
    LLMResponse,
    QuotaInfo,
    estimate_cost,
)
from orion.llm.quota import QuotaTracker

logger = structlog.get_logger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider:
    """OpenRouter pay-per-token provider with budget guard.

    Uses ``httpx.AsyncClient`` directly (not the openai SDK) to
    include required ``HTTP-Referer`` and ``X-Title`` headers.

    Args:
        api_key: OpenRouter API key (``sk-or-...``).
        model: Default model for reasoning requests.
        vision_model: Default model for vision fallback.
        quota_tracker: Shared QuotaTracker with daily budget tracking.
        max_cost_per_task_usd: Maximum per-task spend guard.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b:free",
        vision_model: str = "google/gemma-4-26b-a4b-it:free",
        quota_tracker: QuotaTracker | None = None,
        max_cost_per_task_usd: float = 0.50,
        timeout: float = 90.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._vision_model = vision_model
        self._quota_tracker = quota_tracker or QuotaTracker(
            "openrouter", daily_budget_usd=5.00
        )
        self._max_cost_per_task = max_cost_per_task_usd
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialise the httpx async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=_OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://github.com/orion-agent/orion",
                    "X-Title": "ORION Automation Agent",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        return self._client

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "openrouter"

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
        """Send a chat completion request to OpenRouter.

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
            QuotaExceededError: On 429 or budget exhaustion.
        """
        client = self._get_client()
        used_model = model or self._model
        start = time.monotonic()

        # Pre-check budget
        quota = await self._quota_tracker.get_quota()
        if quota.is_exhausted:
            raise QuotaExceededError(
                "OpenRouter daily budget exhausted",
                provider=self.name,
                daily_budget_remaining_usd=quota.daily_budget_remaining_usd,
            )

        payload: dict[str, Any] = {
            "model": used_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            http_response = await client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "openrouter_timeout",
                provider=self.name,
                model=used_model,
                latency_ms=elapsed,
            )
            raise LLMError(
                f"OpenRouter request timed out after {self._timeout}s",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "openrouter_http_error",
                provider=self.name,
                model=used_model,
                error=str(exc),
                latency_ms=elapsed,
            )
            raise LLMError(
                f"OpenRouter HTTP error: {exc}",
                provider=self.name,
            ) from exc

        elapsed = (time.monotonic() - start) * 1000

        # Handle error status codes
        if http_response.status_code == 429:
            retry_after_str = http_response.headers.get("retry-after")
            retry_after = float(retry_after_str) if retry_after_str else None
            raise QuotaExceededError(
                "OpenRouter rate limit exceeded",
                provider=self.name,
                retry_after_seconds=retry_after,
            )

        if http_response.status_code >= 500:
            raise LLMError(
                f"OpenRouter server error: {http_response.status_code}",
                provider=self.name,
                context={"status_code": http_response.status_code},
            )

        if http_response.status_code >= 400:
            raise LLMError(
                f"OpenRouter client error: {http_response.status_code} - "
                f"{http_response.text[:200]}",
                provider=self.name,
                context={"status_code": http_response.status_code},
            )

        # Parse response
        data = http_response.json()
        raw_headers = {
            k.lower(): v
            for k, v in http_response.headers.items()
            if k.lower().startswith("x-ratelimit")
        }

        # Update quota tracker with rate-limit headers
        if raw_headers:
            await self._quota_tracker.update_from_headers(raw_headers)

        # Extract content and usage
        choices = data.get("choices", [])
        content = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = estimate_cost(used_model, input_tokens, output_tokens)

        # Record spend for budget tracking
        if cost is not None and cost > 0:
            await self._quota_tracker.record_spend(cost)

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
        )

        return result

    async def is_healthy(self) -> bool:
        """Check OpenRouter health via a lightweight API call.

        Returns:
            True if OpenRouter API is reachable.
        """
        try:
            client = self._get_client()
            resp = await client.get("/models")
            return resp.status_code == 200
        except Exception as exc:
            logger.warning(
                "openrouter_health_check_failed",
                provider=self.name,
                error=str(exc),
            )
            return False

    async def remaining_quota(self) -> QuotaInfo | None:
        """Return cached quota info from the QuotaTracker.

        Returns:
            QuotaInfo with rate-limit and budget information.
        """
        return await self._quota_tracker.get_quota()

    async def close(self) -> None:
        """Close the underlying httpx client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
