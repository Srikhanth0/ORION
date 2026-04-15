"""Metrics — Prometheus metrics for ORION observability.

Exposes all counters, histograms, and gauges defined in the
observability contract. Provides a /metrics endpoint.

Module Contract
---------------
- **Inputs**: metric events from agents, tools, LLM calls.
- **Outputs**: Prometheus-compatible /metrics endpoint.

Depends On
----------
- ``prometheus_client`` (Counter, Histogram, Gauge, start_http_server)
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )

    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


def _make_counter(name: str, doc: str, labels: list[str]) -> Counter | None:
    """Create a Prometheus Counter (or None)."""
    if _PROMETHEUS_AVAILABLE:
        return Counter(name, doc, labels)
    return None


def _make_histogram(name: str, doc: str, labels: list[str]) -> Histogram | None:
    """Create a Prometheus Histogram (or None)."""
    if _PROMETHEUS_AVAILABLE:
        return Histogram(name, doc, labels)
    return None


def _make_gauge(name: str, doc: str, labels: list[str]) -> Gauge | None:
    """Create a Prometheus Gauge (or None)."""
    if _PROMETHEUS_AVAILABLE:
        return Gauge(name, doc, labels)
    return None


# ── LLM Metrics ──────────────────────────────────────────

LLM_REQUESTS = _make_counter(
    "orion_llm_requests_total",
    "Total LLM requests by provider, model, status",
    ["provider", "model", "status"],
)

LLM_LATENCY = _make_histogram(
    "orion_llm_latency_seconds",
    "LLM request latency by provider",
    ["provider"],
)

LLM_TOKENS = _make_counter(
    "orion_llm_tokens_total",
    "Total tokens by provider and type (input/output)",
    ["provider", "type"],
)

LLM_COST = _make_counter(
    "orion_llm_cost_usd_total",
    "Total LLM cost in USD by provider",
    ["provider"],
)

PROVIDER_STATUS = _make_gauge(
    "orion_provider_status",
    "Provider availability (0=down, 1=up)",
    ["provider"],
)

LLM_PROVIDER_FAILURES = _make_counter(
    "orion_llm_provider_failures_total",
    "Total LLM provider failures by provider",
    ["provider"],
)

# ── Tool Metrics ─────────────────────────────────────────

TOOL_CALLS = _make_counter(
    "orion_tool_calls_total",
    "Total tool invocations by tool, category, status",
    ["tool", "category", "status"],
)

TOOL_LATENCY = _make_histogram(
    "orion_tool_latency_seconds",
    "Tool invocation latency by tool",
    ["tool"],
)

# ── Task Metrics ─────────────────────────────────────────

TASKS_TOTAL = _make_counter(
    "orion_tasks_total",
    "Total tasks by status (pass/fail/escalate)",
    ["status"],
)

TASK_DURATION = _make_histogram(
    "orion_task_duration_seconds",
    "Task duration in seconds",
    [],
)

SUBTASKS_PARALLEL = _make_gauge(
    "orion_subtasks_parallel_count",
    "Number of subtasks executing in parallel in current batch",
    [],
)

# ── Vision Metrics ───────────────────────────────────────

VISION_API_LATENCY = _make_histogram(
    "orion_vision_api_latency_seconds",
    "Vision API request latency",
    [],
)

# ── Memory Metrics ───────────────────────────────────────

WORKING_MEMORY_TOKENS = _make_gauge(
    "orion_memory_working_tokens",
    "Working memory token utilization per agent",
    ["agent"],
)

LONGTERM_DOCUMENTS = _make_gauge(
    "orion_memory_longterm_documents_total",
    "Total documents in long-term memory",
    [],
)

# ── Circuit Breaker ──────────────────────────────────────

CIRCUIT_BREAKER = _make_gauge(
    "orion_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["provider"],
)


class MetricsServer:
    """Prometheus metrics server management.

    Args:
        port: Port for the /metrics endpoint.
    """

    _running = False

    def __init__(self, port: int = 9091) -> None:
        self._port = port

    def start(self) -> None:
        """Start the Prometheus HTTP server."""
        if not _PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_not_available",
            )
            return

        if MetricsServer._running:
            return

        try:
            start_http_server(self._port)
            MetricsServer._running = True
            logger.info(
                "metrics_server_started",
                port=self._port,
            )
        except Exception as exc:
            logger.error(
                "metrics_server_failed",
                port=self._port,
                error=str(exc),
            )


def record_llm_call(
    provider: str,
    model: str,
    status: str,
    latency_seconds: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Record an LLM call in Prometheus metrics.

    Args:
        provider: Provider name (vllm, groq, openrouter).
        model: Model name.
        status: Call status (success, error, timeout).
        latency_seconds: Call latency.
        input_tokens: Input token count.
        output_tokens: Output token count.
        cost_usd: Estimated cost.
    """
    if LLM_REQUESTS:
        LLM_REQUESTS.labels(provider=provider, model=model, status=status).inc()
    if LLM_LATENCY:
        LLM_LATENCY.labels(provider=provider).observe(latency_seconds)
    if LLM_TOKENS:
        if input_tokens:
            LLM_TOKENS.labels(provider=provider, type="input").inc(input_tokens)
        if output_tokens:
            LLM_TOKENS.labels(provider=provider, type="output").inc(output_tokens)
    if LLM_COST and cost_usd > 0:
        LLM_COST.labels(provider=provider).inc(cost_usd)

    if status == "error" and LLM_PROVIDER_FAILURES:
        LLM_PROVIDER_FAILURES.labels(provider=provider).inc()


def record_tool_call(
    tool: str,
    category: str,
    status: str,
    latency_seconds: float,
) -> None:
    """Record a tool invocation in Prometheus metrics.

    Args:
        tool: Tool name.
        category: Tool category.
        status: Call status (success, error).
        latency_seconds: Invocation latency.
    """
    if TOOL_CALLS:
        TOOL_CALLS.labels(tool=tool, category=category, status=status).inc()
    if TOOL_LATENCY:
        TOOL_LATENCY.labels(tool=tool).observe(latency_seconds)


def record_task(
    status: str,
    duration_seconds: float,
) -> None:
    """Record a task completion.

    Args:
        status: Task status (pass, fail, escalate).
        duration_seconds: Total task duration.
    """
    if TASKS_TOTAL:
        TASKS_TOTAL.labels(status=status).inc()
    if TASK_DURATION:
        TASK_DURATION.observe(duration_seconds)


def record_vision_call(
    latency_seconds: float,
) -> None:
    """Record a vision API call.

    Args:
        latency_seconds: Round-trip latency to the vision server.
    """
    if VISION_API_LATENCY:
        VISION_API_LATENCY.observe(latency_seconds)
