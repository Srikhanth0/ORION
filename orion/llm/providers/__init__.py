"""ORION LLM providers — individual backend wrappers.

Each provider implements the ``LLMProvider`` protocol.

Exports
-------
- ``VLLMProvider`` — local Qwen 2.5 via vLLM (OpenAI-compat)
- ``GroqProvider`` — Groq cloud with quota tracking + vision
- ``OpenRouterProvider`` — OpenRouter with budget guard + vision fallback
- ``LLMProvider`` — the protocol itself (for type hints)
- ``LLMResponse``, ``QuotaInfo``, ``ProviderStatus`` — shared types
"""

from __future__ import annotations

from orion.llm.providers.base import (
    COST_TABLE,
    LLMProvider,
    LLMResponse,
    ProviderStatus,
    QuotaInfo,
    estimate_cost,
)
from orion.llm.providers.groq_provider import GroqProvider
from orion.llm.providers.openrouter_provider import OpenRouterProvider
from orion.llm.providers.vllm_provider import VLLMProvider

__all__: list[str] = [
    "LLMProvider",
    "LLMResponse",
    "QuotaInfo",
    "ProviderStatus",
    "COST_TABLE",
    "estimate_cost",
    "VLLMProvider",
    "GroqProvider",
    "OpenRouterProvider",
]
