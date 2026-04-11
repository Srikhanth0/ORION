"""Adaptive LLM Router with circuit breaker.

Core routing engine that iterates the provider fallback chain in
priority order. Includes circuit breaker logic, structured output
parsing with retry, and observability hooks.

Module Contract
---------------
- **Inputs**: Chat messages, optional model/role override.
- **Outputs**: ``LLMResponse`` from the first healthy+available provider.
- **Failure Modes**: ``AllProvidersExhaustedError`` when every provider fails.

Depends On
----------
- ``orion.llm.providers.base`` (LLMProvider, LLMResponse, ProviderStatus)
- ``orion.llm.health`` (HealthMonitor)
- ``orion.core.exceptions`` (AllProvidersExhaustedError, LLMError, QuotaExceededError)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from orion.core.exceptions import (
    AllProvidersExhaustedError,
    LLMError,
    QuotaExceededError,
)
from orion.llm.health import HealthMonitor
from orion.llm.providers.base import LLMProvider, LLMResponse, ProviderStatus

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ── Structured Output Error ──────────────────────────────────


class StructuredOutputError(LLMError):
    """Raised when structured JSON output parsing fails after retry.

    Args:
        message: Description of the parse failure.
        raw_output: The raw LLM output that could not be parsed.
        schema_name: Name of the Pydantic schema that was expected.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        raw_output: str | None = None,
        schema_name: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.raw_output = raw_output
        self.schema_name = schema_name


# ── Circuit Breaker ──────────────────────────────────────────


class CircuitBreaker:
    """Per-provider circuit breaker.

    Trips after ``failure_threshold`` failures within ``window_seconds``.
    Once tripped, the provider is marked UNAVAILABLE for
    ``recovery_timeout`` seconds regardless of health check results.

    Args:
        failure_threshold: Number of failures before tripping.
        window_seconds: Time window for counting failures.
        recovery_timeout: Seconds to wait before allowing retry.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        window_seconds: float = 60.0,
        recovery_timeout: float = 300.0,
    ) -> None:
        self._threshold = failure_threshold
        self._window = window_seconds
        self._recovery_timeout = recovery_timeout
        self._failures: list[float] = []
        self._tripped_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        """Return True if the circuit breaker is currently tripped."""
        if self._tripped_at is None:
            return False
        return (time.monotonic() - self._tripped_at) < self._recovery_timeout

    async def record_failure(self) -> bool:
        """Record a failure and check if the breaker should trip.

        Returns:
            True if the circuit breaker is now tripped.
        """
        async with self._lock:
            now = time.monotonic()
            self._failures = [
                t for t in self._failures if (now - t) < self._window
            ]
            self._failures.append(now)

            if len(self._failures) >= self._threshold:
                self._tripped_at = now
                logger.warning(
                    "circuit_breaker_tripped",
                    failures_in_window=len(self._failures),
                    recovery_timeout=self._recovery_timeout,
                )
                return True
            return False

    async def record_success(self) -> None:
        """Record a successful call and reset the failure window."""
        async with self._lock:
            self._failures.clear()

    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self._failures.clear()
            self._tripped_at = None


# ── Adaptive LLM Router ─────────────────────────────────────


class AdaptiveLLMRouter:
    """Adaptive LLM router with tiered fallback chain.

    Iterates providers in priority order (vLLM → Groq → OpenRouter).
    For each provider: checks health status → checks quota → attempts
    call. On failure: logs, records circuit breaker failure, tries
    next. Never retries the same provider in one request.

    Args:
        providers: Ordered list of LLM providers (priority order).
        health_monitor: Shared HealthMonitor instance.
        circuit_breaker_config: Config dict with failure_threshold,
            window_seconds, recovery_timeout. Applied per-provider.
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        health_monitor: HealthMonitor,
        circuit_breaker_config: dict[str, Any] | None = None,
    ) -> None:
        self._providers = {p.name: p for p in providers}
        self._provider_order = [p.name for p in providers]
        self._health_monitor = health_monitor

        cb_config = circuit_breaker_config or {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {
            p.name: CircuitBreaker(
                failure_threshold=cb_config.get("failure_threshold", 5),
                window_seconds=cb_config.get("window_seconds", 60.0),
                recovery_timeout=cb_config.get("recovery_timeout", 300.0),
            )
            for p in providers
        }

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
        preferred_provider: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Route a chat request through the fallback chain.

        Args:
            messages: OpenAI-format chat messages.
            model: Override model for this request.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            response_format: Optional response format spec.
            preferred_provider: Skip ahead to this provider if healthy.
            **kwargs: Additional parameters forwarded to provider.

        Returns:
            LLMResponse from the first successful provider.

        Raises:
            AllProvidersExhaustedError: When every provider in the
                chain has failed.
        """
        errors: dict[str, str] = {}
        attempted: list[str] = []

        # Build provider order (preferred first if specified)
        order = list(self._provider_order)
        if preferred_provider and preferred_provider in self._providers:
            order.remove(preferred_provider)
            order.insert(0, preferred_provider)

        for provider_name in order:
            provider = self._providers[provider_name]
            cb = self._circuit_breakers[provider_name]

            # Check circuit breaker
            if cb.is_open:
                errors[provider_name] = "circuit breaker open"
                logger.debug(
                    "provider_skipped_circuit_breaker",
                    provider=provider_name,
                )
                continue

            # Check health status
            status = await self._health_monitor.get_status(provider_name)
            if status == ProviderStatus.UNAVAILABLE:
                errors[provider_name] = "unavailable (health check)"
                logger.debug(
                    "provider_skipped_unavailable",
                    provider=provider_name,
                )
                continue

            # Check quota (skip if exhausted or critically low)
            try:
                quota = await provider.remaining_quota()
                if quota is not None and quota.is_exhausted:
                    errors[provider_name] = "quota exhausted"
                    logger.debug(
                        "provider_skipped_quota_exhausted",
                        provider=provider_name,
                    )
                    continue
            except Exception:
                pass  # Quota check failure is non-fatal

            # Attempt the call
            attempted.append(provider_name)
            try:
                response = await provider.chat(
                    messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    **kwargs,
                )

                # Success — reset circuit breaker and mark healthy
                await cb.record_success()
                await self._health_monitor.set_status(
                    provider_name, ProviderStatus.HEALTHY
                )

                return response

            except QuotaExceededError as exc:
                errors[provider_name] = f"quota exceeded: {exc}"
                logger.warning(
                    "provider_quota_exceeded",
                    provider=provider_name,
                    error=str(exc),
                )
                continue  # Don't trip circuit breaker on quota errors

            except (LLMError, TimeoutError, Exception) as exc:
                errors[provider_name] = str(exc)
                tripped = await cb.record_failure()

                if tripped:
                    await self._health_monitor.set_status(
                        provider_name, ProviderStatus.UNAVAILABLE
                    )

                logger.warning(
                    "provider_call_failed",
                    provider=provider_name,
                    error=str(exc),
                    circuit_breaker_tripped=tripped,
                )
                continue

        # All providers failed
        raise AllProvidersExhaustedError(
            f"All {len(self._provider_order)} providers failed: "
            + "; ".join(f"{k}: {v}" for k, v in errors.items()),
            attempted_providers=attempted,
            errors=errors,
        )

    async def chat_structured(
        self,
        messages: list[dict[str, Any]],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        preferred_provider: str | None = None,
        **kwargs: Any,
    ) -> T:
        """Chat with structured JSON output parsed into a Pydantic model.

        Appends JSON-mode instruction to the system prompt, requests
        ``response_format={"type": "json_object"}``, and parses the
        result with ``schema.model_validate_json()``.

        On parse failure: retries once with error feedback appended.
        On second failure: raises ``StructuredOutputError``.

        Args:
            messages: OpenAI-format chat messages.
            schema: Pydantic model class to parse the response into.
            model: Override model for this request.
            temperature: Sampling temperature (lower for structured output).
            max_tokens: Maximum output tokens.
            preferred_provider: Skip ahead to this provider if healthy.
            **kwargs: Additional parameters.

        Returns:
            Parsed Pydantic model instance.

        Raises:
            StructuredOutputError: After two parse failures.
            AllProvidersExhaustedError: If no provider can serve.
        """
        # Build JSON instruction
        schema_json = schema.model_json_schema()
        json_instruction = (
            "You MUST respond with ONLY a valid JSON object matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            "Do NOT include any text outside the JSON object."
        )

        # Inject or append to system message
        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + "\n\n" + json_instruction,
            }
        else:
            augmented.insert(0, {"role": "system", "content": json_instruction})

        # First attempt
        response = await self.chat(
            augmented,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            preferred_provider=preferred_provider,
            **kwargs,
        )

        parse_error: ValidationError | ValueError | None = None
        try:
            return schema.model_validate_json(response.content)
        except (ValidationError, ValueError) as first_error:
            parse_error = first_error
            logger.warning(
                "structured_output_parse_failed",
                schema=schema.__name__,
                error=str(first_error),
                attempt=1,
            )

        # Retry with error feedback
        augmented.append({
            "role": "assistant",
            "content": response.content,
        })
        augmented.append({
            "role": "user",
            "content": (
                f"Your response was not valid JSON matching the schema. "
                f"Error: {parse_error}\n\n"
                f"Please fix the JSON and respond again with ONLY the "
                f"corrected JSON object."
            ),
        })

        response2 = await self.chat(
            augmented,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            preferred_provider=preferred_provider,
            **kwargs,
        )

        try:
            return schema.model_validate_json(response2.content)
        except (ValidationError, ValueError) as second_error:
            logger.error(
                "structured_output_parse_failed_final",
                schema=schema.__name__,
                error=str(second_error),
                attempt=2,
            )
            raise StructuredOutputError(
                f"Failed to parse structured output after 2 attempts: {second_error}",
                raw_output=response2.content,
                schema_name=schema.__name__,
            ) from second_error
