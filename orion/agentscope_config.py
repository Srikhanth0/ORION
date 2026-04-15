"""AgentScope initialization bridge for ORION.

Compatible with AgentScope v1.0+ where ``agentscope.init()`` no longer
accepts a ``model_configs`` parameter.  Models are now instantiated
directly via their class constructors (e.g. ``OpenAIChatModel``).

Module Contract
---------------
- ``init_agentscope()`` — call once at startup (idempotent).
- ``build_model(provider)`` — factory that returns a ready ChatModel.

Depends On
----------
- ``agentscope`` >= 1.0.18
- ``configs/llm/router.yaml`` (optional, for role mapping)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_ROUTER_CONFIG = "configs/llm/router.yaml"
_initialized = False


# ── Bootstrap ────────────────────────────────────────────────────────────────


def _load_env() -> None:
    """Load .env once so API keys are available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


def init_agentscope(
    router_config_path: str = _DEFAULT_ROUTER_CONFIG,
) -> None:
    """Initialize AgentScope framework (idempotent).

    In AgentScope v1.0+ ``init()`` only accepts project / logging args.
    Model registration is handled by direct instantiation in
    ``build_model()``.

    Args:
        router_config_path: Path to router YAML (logged, not used for
            model registration).
    """
    global _initialized
    if _initialized:
        return

    _load_env()

    import agentscope

    # v1.0+ signature: init(project, name, run_id, logging_path, …)
    agentscope.init(project="ORION")

    cfg_path = Path(router_config_path)
    if cfg_path.exists():
        try:
            import yaml

            cfg = (
                yaml.safe_load(
                    cfg_path.read_text(encoding="utf-8"),
                )
                or {}
            )
            roles = cfg.get("roles", {})
            logger.info(
                "agentscope_initialized",
                roles=list(roles.keys()),
            )
        except Exception as exc:
            logger.warning("agentscope_config_read_error", error=str(exc))
    else:
        logger.warning(
            "agentscope_config_not_found",
            path=router_config_path,
        )

    _initialized = True


# ── Model factory ────────────────────────────────────────────────────────────


def build_model(provider: str | None = None) -> Any:
    """Instantiate an AgentScope chat model for the requested provider.

    Falls back automatically:  groq → openrouter → error.

    Args:
        provider: ``"groq"`` | ``"openrouter"`` | ``"vllm"`` | ``None``
            (auto-detect from env vars).

    Returns:
        ``OpenAIChatModel`` instance ready for ``__call__``.

    Raises:
        ValueError: If the provider is unknown or no API key is set.
    """
    _load_env()

    from agentscope.model import OpenAIChatModel

    groq_key = os.environ.get("GROQ_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    vllm_url = os.environ.get("VLLM_BASE_URL", "")

    # ── Auto-detect ──────────────────────────────────────────────
    if provider is None:
        provider = os.environ.get("ORION_LLM_PROVIDER")

    if provider is None:
        if groq_key:
            provider = "groq"
        elif openrouter_key:
            provider = "openrouter"
        elif vllm_url:
            provider = "vllm"
        else:
            raise ValueError(
                "No LLM provider configured. Set GROQ_API_KEY or OPENROUTER_API_KEY in .env"
            )

    # ── Groq ─────────────────────────────────────────────────────
    if provider == "groq":
        os.environ["OPENAI_API_KEY"] = groq_key
        os.environ["OPENAI_BASE_URL"] = "https://api.groq.com/openai/v1"
        return OpenAIChatModel(
            config_name="orion_groq",
            model_name=os.environ.get(
                "GROQ_TEXT_MODEL",
                "llama-3.3-70b-versatile",
            ),
            api_key=groq_key,
            client_kwargs={"timeout": 60.0, "max_retries": 2},
        )

    # ── OpenRouter ───────────────────────────────────────────────
    if provider == "openrouter":
        os.environ["OPENAI_API_KEY"] = openrouter_key
        os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
        return OpenAIChatModel(
            config_name="orion_openrouter",
            model_name=os.environ.get(
                "OPENROUTER_DEFAULT_MODEL",
                "google/gemma-2-9b-it:free",
            ),
            api_key=openrouter_key,
            client_kwargs={"timeout": 60.0, "max_retries": 2},
        )

    # ── vLLM (self-hosted / Colab) ───────────────────────────────
    if provider == "vllm":
        base = vllm_url.rstrip("/")
        os.environ["OPENAI_API_KEY"] = "not-needed"
        os.environ["OPENAI_BASE_URL"] = f"{base}/v1"
        return OpenAIChatModel(
            config_name="orion_vllm",
            model_name=os.environ.get(
                "VLLM_MODEL",
                "Qwen/Qwen3-VL-4B-Instruct",
            ),
            api_key="not-needed",
            client_kwargs={"timeout": 60.0, "max_retries": 2},
        )

    raise ValueError(f"Unknown LLM provider: {provider!r}")
