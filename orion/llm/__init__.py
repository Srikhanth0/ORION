"""ORION LLM — Adaptive router and provider wrappers.

Public API Surface
------------------
- ``AdaptiveLLMRouter`` — main routing engine
- ``LLMResponse`` — normalised response container
- ``StructuredOutputError`` — structured output parse failure
- ``LLMProvider`` — provider protocol (for typing)
- ``ProviderStatus`` — health status enum
- ``QuotaInfo`` — quota snapshot
- ``CircuitBreaker`` — per-provider circuit breaker
- ``HealthMonitor`` — background health poller
- ``QuotaTracker`` — rate-limit header tracker

Providers (import from ``orion.llm.providers``):
- ``VLLMProvider`` — local Qwen 2.5 via vLLM
- ``GroqProvider`` — Groq free tier with quota tracking
- ``OpenRouterProvider`` — OpenRouter with budget guard
"""
from __future__ import annotations

from orion.llm.health import HealthMonitor
from orion.llm.providers.base import (
    COST_TABLE,
    LLMProvider,
    LLMResponse,
    ProviderStatus,
    QuotaInfo,
    estimate_cost,
)
from orion.llm.quota import QuotaTracker
from orion.llm.router import AdaptiveLLMRouter, CircuitBreaker, StructuredOutputError

__all__: list[str] = [
    # Router
    "AdaptiveLLMRouter",
    "CircuitBreaker",
    "StructuredOutputError",
    # Response types
    "LLMResponse",
    "QuotaInfo",
    "ProviderStatus",
    # Protocol
    "LLMProvider",
    # Infrastructure
    "HealthMonitor",
    "QuotaTracker",
    # Utilities
    "COST_TABLE",
    "estimate_cost",
]
