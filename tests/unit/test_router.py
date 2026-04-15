"""Unit tests for orion.llm — AdaptiveLLMRouter, CircuitBreaker, providers.

Tests NEVER call real APIs. All provider calls are mocked at the
protocol level using pytest-mock and AsyncMock.

Test Scenarios
--------------
1. All providers healthy → vLLM (priority 1) is used
2. vLLM unhealthy → Groq (priority 2) is used
3. vLLM + Groq exhausted → OpenRouter (priority 3) is used
4. All providers fail → AllProvidersExhaustedError raised
5. Groq quota at 95% → skipped, OpenRouter used
6. Structured output parse fails → retry with error feedback
7. Circuit breaker trips after 5 failures in 60s
8. Preferred provider override
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from orion.core.exceptions import (
    AllProvidersExhaustedError,
    LLMError,
    QuotaExceededError,
)
from orion.llm.health import HealthMonitor
from orion.llm.providers.base import (
    LLMResponse,
    ProviderStatus,
    QuotaInfo,
    estimate_cost,
)
from orion.llm.quota import QuotaTracker
from orion.llm.router import AdaptiveLLMRouter, CircuitBreaker, StructuredOutputError

# ── Test Helpers & Fixtures ──────────────────────────────────


def _make_response(
    content: str = "Hello!",
    provider: str = "vllm",
    model: str = "qwen2.5-72b",
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> LLMResponse:
    """Create a mock LLMResponse for testing."""
    return LLMResponse(
        content=content,
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=100.0,
        cost_usd=0.0,
    )


class MockProvider:
    """A mock LLM provider implementing the LLMProvider protocol.

    Args:
        provider_name: Name of this provider.
        healthy: Whether is_healthy() returns True.
        quota: QuotaInfo to return from remaining_quota().
        response: LLMResponse to return from chat().
        error: If set, chat() raises this exception instead.
    """

    def __init__(
        self,
        provider_name: str = "mock",
        healthy: bool = True,
        quota: QuotaInfo | None = None,
        response: LLMResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._name = provider_name
        self._healthy = healthy
        self._quota = quota
        self._response = response or _make_response(provider=provider_name)
        self._error = error
        self.call_count = 0

    @property
    def name(self) -> str:
        return self._name

    async def chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> LLMResponse:
        self.call_count += 1
        if self._error is not None:
            raise self._error
        return self._response

    async def is_healthy(self) -> bool:
        return self._healthy

    async def remaining_quota(self) -> QuotaInfo | None:
        return self._quota


def _make_providers(
    vllm_healthy: bool = True,
    groq_healthy: bool = True,
    openrouter_healthy: bool = True,
    vllm_error: Exception | None = None,
    groq_error: Exception | None = None,
    openrouter_error: Exception | None = None,
    groq_quota: QuotaInfo | None = None,
    openrouter_quota: QuotaInfo | None = None,
) -> list[MockProvider]:
    """Create a standard 3-provider chain for testing."""
    return [
        MockProvider(
            provider_name="vllm",
            healthy=vllm_healthy,
            response=_make_response(provider="vllm"),
            error=vllm_error,
        ),
        MockProvider(
            provider_name="groq",
            healthy=groq_healthy,
            response=_make_response(provider="groq", model="llama-3.3-70b"),
            error=groq_error,
            quota=groq_quota,
        ),
        MockProvider(
            provider_name="openrouter",
            healthy=openrouter_healthy,
            response=_make_response(provider="openrouter", model="gpt-oss-120b"),
            error=openrouter_error,
            quota=openrouter_quota,
        ),
    ]


async def _make_router(
    providers: list[MockProvider],
    cb_config: dict[str, Any] | None = None,
) -> AdaptiveLLMRouter:
    """Create an AdaptiveLLMRouter with pre-configured health statuses."""
    health = HealthMonitor(providers)  # type: ignore[arg-type]

    # Set initial health statuses based on provider._healthy
    for p in providers:
        status = ProviderStatus.HEALTHY if p._healthy else ProviderStatus.UNAVAILABLE
        await health.set_status(p.name, status)

    return AdaptiveLLMRouter(
        providers=providers,  # type: ignore[arg-type]
        health_monitor=health,
        circuit_breaker_config=cb_config,
    )


_MESSAGES = [{"role": "user", "content": "Hello"}]


# ── Scenario 1: All healthy → vLLM used ─────────────────────


class TestAllHealthy:
    """When all providers are healthy, the highest priority (vLLM) is used."""

    @pytest.mark.asyncio
    async def test_vllm_is_first_choice(self) -> None:
        providers = _make_providers()
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        assert response.provider == "vllm"
        assert providers[0].call_count == 1
        assert providers[1].call_count == 0
        assert providers[2].call_count == 0


# ── Scenario 2: vLLM unhealthy → Groq used ──────────────────


class TestVLLMUnhealthy:
    """When vLLM is unhealthy, Groq (priority 2) is used."""

    @pytest.mark.asyncio
    async def test_groq_fallback(self) -> None:
        providers = _make_providers(vllm_healthy=False)
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        assert response.provider == "groq"
        assert providers[0].call_count == 0
        assert providers[1].call_count == 1


# ── Scenario 3: vLLM + Groq exhausted → OpenRouter ──────────


class TestVLLMAndGroqExhausted:
    """When vLLM and Groq both fail, OpenRouter (priority 3) is used."""

    @pytest.mark.asyncio
    async def test_openrouter_fallback(self) -> None:
        providers = _make_providers(
            vllm_healthy=False,
            groq_healthy=False,
        )
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        assert response.provider == "openrouter"
        assert providers[2].call_count == 1

    @pytest.mark.asyncio
    async def test_openrouter_fallback_on_errors(self) -> None:
        """vLLM and Groq healthy but fail on call → OpenRouter used."""
        providers = _make_providers(
            vllm_error=LLMError("vllm down", provider="vllm"),
            groq_error=LLMError("groq error", provider="groq"),
        )
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        assert response.provider == "openrouter"
        assert providers[0].call_count == 1
        assert providers[1].call_count == 1
        assert providers[2].call_count == 1


# ── Scenario 4: All fail → AllProvidersExhaustedError ────────


class TestAllProvidersFail:
    """When every provider fails, AllProvidersExhaustedError is raised."""

    @pytest.mark.asyncio
    async def test_all_exhausted_error(self) -> None:
        providers = _make_providers(
            vllm_error=LLMError("vllm fail", provider="vllm"),
            groq_error=LLMError("groq fail", provider="groq"),
            openrouter_error=LLMError("openrouter fail", provider="openrouter"),
        )
        router = await _make_router(providers)

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.chat(_MESSAGES)

        exc = exc_info.value
        assert len(exc.attempted_providers) == 3
        assert "vllm" in exc.errors
        assert "groq" in exc.errors
        assert "openrouter" in exc.errors

    @pytest.mark.asyncio
    async def test_all_unhealthy(self) -> None:
        """All providers unhealthy → error with no attempts."""
        providers = _make_providers(
            vllm_healthy=False,
            groq_healthy=False,
            openrouter_healthy=False,
        )
        router = await _make_router(providers)

        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.chat(_MESSAGES)

        assert len(exc_info.value.attempted_providers) == 0


# ── Scenario 5: Groq quota low → skipped ────────────────────


class TestQuotaAware:
    """When Groq quota is exhausted, it's skipped in favor of OpenRouter."""

    @pytest.mark.asyncio
    async def test_quota_exhausted_skip(self) -> None:
        exhausted_quota = QuotaInfo(remaining_requests=0, remaining_tokens=0)
        providers = _make_providers(
            vllm_healthy=False,
            groq_quota=exhausted_quota,
        )
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        # Groq skipped due to exhausted quota, OpenRouter used
        assert response.provider == "openrouter"
        assert providers[1].call_count == 0

    @pytest.mark.asyncio
    async def test_quota_exceeded_error_triggers_fallback(self) -> None:
        """Groq raises QuotaExceededError → falls through to OpenRouter."""
        providers = _make_providers(
            vllm_healthy=False,
            groq_error=QuotaExceededError(
                "rate limited", provider="groq", retry_after_seconds=30.0
            ),
        )
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES)

        assert response.provider == "openrouter"
        assert providers[1].call_count == 1  # Attempted but failed


# ── Scenario 6: Structured output parse failure → retry ──────


class TestStructuredOutput:
    """chat_structured retries once on parse failure, then raises."""

    class _TestSchema(BaseModel):
        name: str
        value: int

    @pytest.mark.asyncio
    async def test_structured_output_success(self) -> None:
        """Valid JSON response is parsed correctly."""
        valid_json = '{"name": "test", "value": 42}'
        providers = [
            MockProvider(
                provider_name="vllm",
                response=_make_response(content=valid_json),
            )
        ]
        router = await _make_router(providers)

        result = await router.chat_structured(_MESSAGES, self._TestSchema)

        assert result.name == "test"
        assert result.value == 42

    @pytest.mark.asyncio
    async def test_structured_output_retry_on_parse_error(self) -> None:
        """Invalid JSON on first try → retry with error feedback → success."""
        call_count = 0

        class RetryProvider(MockProvider):
            async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First attempt: invalid JSON
                    return _make_response(content="not valid json {{{")
                # Second attempt: valid JSON
                return _make_response(content='{"name": "fixed", "value": 99}')

        providers = [RetryProvider(provider_name="vllm")]
        router = await _make_router(providers)

        result = await router.chat_structured(_MESSAGES, self._TestSchema)

        assert result.name == "fixed"
        assert result.value == 99
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_structured_output_both_attempts_fail(self) -> None:
        """Both parse attempts fail → StructuredOutputError."""
        providers = [
            MockProvider(
                provider_name="vllm",
                response=_make_response(content="not json at all"),
            )
        ]
        router = await _make_router(providers)

        with pytest.raises(StructuredOutputError) as exc_info:
            await router.chat_structured(_MESSAGES, self._TestSchema)

        assert exc_info.value.schema_name == "_TestSchema"
        assert exc_info.value.raw_output == "not json at all"


# ── Scenario 7: Circuit breaker trips ────────────────────────


class TestCircuitBreaker:
    """Circuit breaker trips after failure_threshold failures in window."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_trips(self) -> None:
        """5 failures in 60s trips the breaker."""
        cb = CircuitBreaker(failure_threshold=5, window_seconds=60.0, recovery_timeout=300.0)

        assert not cb.is_open

        for _ in range(4):
            tripped = await cb.record_failure()
            assert not tripped

        tripped = await cb.record_failure()
        assert tripped
        assert cb.is_open

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self) -> None:
        """A success resets the failure count."""
        cb = CircuitBreaker(failure_threshold=3, window_seconds=60.0)

        await cb.record_failure()
        await cb.record_failure()
        await cb.record_success()

        # After success, should need 3 more failures to trip
        tripped = await cb.record_failure()
        assert not tripped

    @pytest.mark.asyncio
    async def test_circuit_breaker_manual_reset(self) -> None:
        """Manual reset clears the tripped state."""
        cb = CircuitBreaker(failure_threshold=2, window_seconds=60.0)

        await cb.record_failure()
        await cb.record_failure()
        assert cb.is_open

        await cb.reset()
        assert not cb.is_open

    @pytest.mark.asyncio
    async def test_router_uses_circuit_breaker(self) -> None:
        """Router skips providers with tripped circuit breakers."""
        providers = _make_providers(
            vllm_error=LLMError("fail", provider="vllm"),
        )
        router = await _make_router(
            providers,
            cb_config={"failure_threshold": 2, "window_seconds": 60.0},
        )

        # First call: vLLM fails, falls to Groq
        r1 = await router.chat(_MESSAGES)
        assert r1.provider == "groq"

        # Second call: vLLM fails again, breaker trips
        r2 = await router.chat(_MESSAGES)
        assert r2.provider == "groq"

        # Third call: vLLM skipped (breaker open), goes to Groq
        r3 = await router.chat(_MESSAGES)
        assert r3.provider == "groq"
        # vLLM should have been called only twice (before breaker tripped)
        assert providers[0].call_count == 2


# ── Scenario 8: Preferred provider override ──────────────────


class TestPreferredProvider:
    """preferred_provider overrides the default priority order."""

    @pytest.mark.asyncio
    async def test_preferred_provider_used(self) -> None:
        """Preferred provider is tried first even if not default priority."""
        providers = _make_providers()
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES, preferred_provider="openrouter")

        assert response.provider == "openrouter"
        assert providers[0].call_count == 0  # vLLM not tried
        assert providers[2].call_count == 1

    @pytest.mark.asyncio
    async def test_preferred_provider_fails_then_fallback(self) -> None:
        """If preferred provider fails, falls back to normal chain."""
        providers = _make_providers(
            openrouter_error=LLMError("openrouter down", provider="openrouter"),
        )
        router = await _make_router(providers)

        response = await router.chat(_MESSAGES, preferred_provider="openrouter")

        # OpenRouter tried first but failed, then vLLM succeeds
        assert response.provider == "vllm"
        assert providers[2].call_count == 1  # OpenRouter tried
        assert providers[0].call_count == 1  # vLLM succeeded


# ── Cost Estimation Tests ────────────────────────────────────


class TestCostEstimation:
    """Tests for the cost estimation utility."""

    def test_free_model_cost(self) -> None:
        """Free-tier models return 0.0 cost."""
        cost = estimate_cost("llama-3.3-70b-versatile", 1000, 500)
        assert cost == 0.0

    def test_paid_model_cost(self) -> None:
        """Paid models calculate cost correctly."""
        cost = estimate_cost("deepseek/deepseek-chat", 1000, 1000)
        assert cost is not None
        assert cost > 0.0
        # 1k input * 0.00014 + 1k output * 0.00028 = 0.00042
        assert abs(cost - 0.00042) < 1e-6

    def test_unknown_model_cost(self) -> None:
        """Unknown models return None."""
        cost = estimate_cost("unknown-model-xyz", 1000, 500)
        assert cost is None


# ── QuotaInfo Tests ──────────────────────────────────────────


class TestQuotaInfo:
    """Tests for QuotaInfo helper properties."""

    def test_quota_not_exhausted(self) -> None:
        q = QuotaInfo(remaining_requests=10, remaining_tokens=5000)
        assert not q.is_exhausted
        assert not q.is_low

    def test_quota_exhausted_by_requests(self) -> None:
        q = QuotaInfo(remaining_requests=0)
        assert q.is_exhausted

    def test_quota_exhausted_by_tokens(self) -> None:
        q = QuotaInfo(remaining_tokens=0)
        assert q.is_exhausted

    def test_quota_exhausted_by_budget(self) -> None:
        q = QuotaInfo(daily_budget_remaining_usd=0.0)
        assert q.is_exhausted

    def test_quota_low_requests(self) -> None:
        q = QuotaInfo(remaining_requests=2)
        assert q.is_low

    def test_quota_low_tokens(self) -> None:
        q = QuotaInfo(remaining_tokens=500)
        assert q.is_low

    def test_quota_none_fields(self) -> None:
        """All None fields means not exhausted and not low."""
        q = QuotaInfo()
        assert not q.is_exhausted
        assert not q.is_low


# ── QuotaTracker Tests ───────────────────────────────────────


class TestQuotaTracker:
    """Tests for the QuotaTracker rate-limit header parser."""

    @pytest.mark.asyncio
    async def test_update_from_headers(self) -> None:
        tracker = QuotaTracker("groq")
        await tracker.update_from_headers(
            {
                "x-ratelimit-remaining-requests": "25",
                "x-ratelimit-remaining-tokens": "10000",
            }
        )

        quota = await tracker.get_quota()
        assert quota.remaining_requests == 25
        assert quota.remaining_tokens == 10000

    @pytest.mark.asyncio
    async def test_budget_tracking(self) -> None:
        tracker = QuotaTracker("openrouter", daily_budget_usd=5.00)

        await tracker.record_spend(1.50)
        quota = await tracker.get_quota()
        assert quota.daily_budget_remaining_usd is not None
        assert abs(quota.daily_budget_remaining_usd - 3.50) < 0.01

    @pytest.mark.asyncio
    async def test_reset(self) -> None:
        tracker = QuotaTracker("groq")
        await tracker.update_from_headers(
            {
                "x-ratelimit-remaining-requests": "0",
            }
        )
        quota = await tracker.get_quota()
        assert quota.remaining_requests == 0

        await tracker.reset()
        quota = await tracker.get_quota()
        assert quota.remaining_requests is None


# ── HealthMonitor Tests ──────────────────────────────────────


class TestHealthMonitor:
    """Tests for the HealthMonitor status tracking."""

    @pytest.mark.asyncio
    async def test_initial_status_unavailable(self) -> None:
        providers = _make_providers()
        monitor = HealthMonitor(providers)  # type: ignore[arg-type]

        status = await monitor.get_status("vllm")
        assert status == ProviderStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_initial_check_sets_healthy(self) -> None:
        providers = _make_providers()
        monitor = HealthMonitor(providers)  # type: ignore[arg-type]

        await monitor.run_initial_check()
        status = await monitor.get_status("vllm")
        assert status == ProviderStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_provider_marked_degraded(self) -> None:
        providers = _make_providers(groq_healthy=False)
        monitor = HealthMonitor(providers)  # type: ignore[arg-type]

        await monitor.run_initial_check()
        status = await monitor.get_status("groq")
        assert status in (ProviderStatus.DEGRADED, ProviderStatus.UNAVAILABLE)

    @pytest.mark.asyncio
    async def test_manual_status_override(self) -> None:
        providers = _make_providers()
        monitor = HealthMonitor(providers)  # type: ignore[arg-type]

        await monitor.set_status("vllm", ProviderStatus.UNAVAILABLE)
        status = await monitor.get_status("vllm")
        assert status == ProviderStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_get_all_statuses(self) -> None:
        providers = _make_providers()
        monitor = HealthMonitor(providers)  # type: ignore[arg-type]
        await monitor.run_initial_check()

        statuses = await monitor.get_all_statuses()
        assert len(statuses) == 3
        assert "vllm" in statuses
        assert "groq" in statuses
        assert "openrouter" in statuses


# ── LLMResponse Tests ────────────────────────────────────────


class TestLLMResponse:
    """Tests for the LLMResponse dataclass."""

    def test_response_creation(self) -> None:
        r = _make_response()
        assert r.content == "Hello!"
        assert r.provider == "vllm"
        assert r.cost_usd == 0.0

    def test_response_is_frozen(self) -> None:
        r = _make_response()
        with pytest.raises(AttributeError):
            r.content = "changed"  # type: ignore[misc]
