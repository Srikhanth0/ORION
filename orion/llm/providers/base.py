"""LLM provider protocol, response containers, and shared types.

Defines the contract that all LLM providers (vLLM, Groq, OpenRouter)
must implement. Framework-agnostic — no provider SDKs imported here.

Module Contract
---------------
- **Outputs**: ``LLMProvider`` protocol, ``LLMResponse``, ``QuotaInfo``,
  ``ProviderStatus`` enum, ``COST_TABLE``.
- **Must NOT Know About**: Router logic, health monitoring, circuit
  breakers, or specific provider implementations.

Depends On
----------
- Python stdlib + ``pydantic`` only.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

# ── Provider Status ──────────────────────────────────────────


class ProviderStatus(enum.StrEnum):
    """Health status of an LLM provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


# ── Response Container ───────────────────────────────────────


@dataclass(frozen=True)
class LLMResponse:
    """Normalised response from any LLM provider.

    Every provider must return this exact type from ``chat()``.

    Attributes:
        content: The LLM-generated text.
        provider: Provider name that produced this response (vllm/groq/openrouter).
        model: The model identifier used.
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        latency_ms: Wall-clock latency in milliseconds.
        cost_usd: Estimated cost in USD (None for free/local providers).
        raw_headers: Raw response headers for quota tracking (optional).
    """

    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    raw_headers: dict[str, str] = field(default_factory=dict)


# ── Quota Info ───────────────────────────────────────────────


@dataclass(frozen=True)
class QuotaInfo:
    """Snapshot of a provider's remaining quota.

    Attributes:
        remaining_requests: Remaining API requests (from rate-limit headers).
        remaining_tokens: Remaining tokens (from rate-limit headers).
        daily_budget_remaining_usd: Remaining budget for pay-per-token providers.
        reset_at_epoch: Unix timestamp when quota resets (if known).
    """

    remaining_requests: int | None = None
    remaining_tokens: int | None = None
    daily_budget_remaining_usd: float | None = None
    reset_at_epoch: float | None = None

    @property
    def is_exhausted(self) -> bool:
        """Return True if any known quota metric is at zero."""
        if self.remaining_requests is not None and self.remaining_requests <= 0:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens <= 0:
            return True
        return (
            self.daily_budget_remaining_usd is not None and self.daily_budget_remaining_usd <= 0.0
        )

    @property
    def is_low(self) -> bool:
        """Return True if quota is critically low (< 5% remaining)."""
        if self.remaining_requests is not None and self.remaining_requests < 3:
            return True
        if self.remaining_tokens is not None and self.remaining_tokens < 1000:
            return True
        return (
            self.daily_budget_remaining_usd is not None and self.daily_budget_remaining_usd < 0.10
        )


# ── LLM Provider Protocol ───────────────────────────────────

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Contract that all LLM providers must implement.

    Providers are stateless except for optional quota tracking.
    The router calls these methods in priority order during
    the fallback chain.
    """

    @property
    def name(self) -> str:
        """Unique provider identifier (e.g., 'vllm', 'groq', 'openrouter')."""
        ...

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
        """Send a chat completion request to the provider.

        Args:
            messages: OpenAI-format chat messages.
            model: Override the default model for this request.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional response format (e.g., {"type": "json_object"}).
            **kwargs: Additional provider-specific parameters.

        Returns:
            Normalised LLMResponse.

        Raises:
            LLMError: On provider-level failures.
            QuotaExceededError: When rate limit is hit.
        """
        ...

    async def is_healthy(self) -> bool:
        """Lightweight health check (e.g., models.list or /health).

        Returns:
            True if the provider is reachable and operational.
        """
        ...

    async def remaining_quota(self) -> QuotaInfo | None:
        """Return current quota snapshot, or None if not tracked.

        Returns:
            QuotaInfo with remaining requests/tokens/budget, or None.
        """
        ...


# ── Cost Table ───────────────────────────────────────────────

# Mapping: model_name → (input_price_per_1k_tokens, output_price_per_1k_tokens) in USD
# Free-tier models have (0.0, 0.0). Updated as of 2025-Q4.
COST_TABLE: dict[str, tuple[float, float]] = {
    # vLLM (local, zero-cost)
    "qwen2.5-72b": (0.0, 0.0),
    "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4": (0.0, 0.0),
    # Groq (free tier)
    "llama-3.3-70b-versatile": (0.0, 0.0),
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.0, 0.0),
    # OpenRouter (free tier models)
    "openai/gpt-oss-120b:free": (0.0, 0.0),
    "google/gemma-4-26b-a4b-it:free": (0.0, 0.0),
    # OpenRouter (paid models, if used)
    "deepseek/deepseek-chat": (0.00014, 0.00028),
    "anthropic/claude-haiku-4-5": (0.0008, 0.004),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost for a request based on the cost table.

    Args:
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, or None if model is not in cost table.
    """
    prices = COST_TABLE.get(model)
    if prices is None:
        return None
    input_price, output_price = prices
    return (input_tokens / 1000 * input_price) + (output_tokens / 1000 * output_price)
