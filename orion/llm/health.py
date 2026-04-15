"""Health monitoring for LLM providers.

Runs a background ``asyncio.Task`` that polls each provider's
``is_healthy()`` at configurable intervals. Supports exponential
backoff on repeated failures and emits Prometheus-compatible
status gauges.

Module Contract
---------------
- **Inputs**: List of ``LLMProvider`` instances.
- **Outputs**: Thread-safe dict of ``{provider_name: ProviderStatus}``.
- **Thread Safety**: ``asyncio.Lock``-guarded status dict.

Depends On
----------
- ``orion.llm.providers.base`` (LLMProvider, ProviderStatus)
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import structlog

from orion.llm.providers.base import LLMProvider, ProviderStatus

logger = structlog.get_logger(__name__)

# Exponential backoff intervals (seconds) on repeated failures
_BACKOFF_INTERVALS = [30, 60, 120, 300]


class HealthMonitor:
    """Background health monitor for LLM providers.

    Polls each provider's ``is_healthy()`` method at regular intervals.
    On repeated failures, applies exponential backoff. Provider status
    is stored in a thread-safe dict guarded by ``asyncio.Lock``.

    Args:
        providers: List of LLM provider instances to monitor.
        poll_interval: Base polling interval in seconds (default: 30).
    """

    def __init__(
        self,
        providers: list[LLMProvider],
        poll_interval: float = 30.0,
    ) -> None:
        self._providers = {p.name: p for p in providers}
        self._poll_interval = poll_interval
        self._lock = asyncio.Lock()
        self._status: dict[str, ProviderStatus] = {
            p.name: ProviderStatus.UNAVAILABLE for p in providers
        }
        self._failure_counts: dict[str, int] = {p.name: 0 for p in providers}
        self._task: asyncio.Task[None] | None = None

    async def get_status(self, provider_name: str) -> ProviderStatus:
        """Get current status of a provider.

        Args:
            provider_name: Provider identifier.

        Returns:
            Current ProviderStatus.
        """
        async with self._lock:
            return self._status.get(provider_name, ProviderStatus.UNAVAILABLE)

    async def get_all_statuses(self) -> dict[str, ProviderStatus]:
        """Get status dict for all monitored providers.

        Returns:
            Copy of the status dict.
        """
        async with self._lock:
            return dict(self._status)

    async def set_status(self, provider_name: str, status: ProviderStatus) -> None:
        """Manually override a provider's status.

        Used by the circuit breaker to force UNAVAILABLE status.

        Args:
            provider_name: Provider identifier.
            status: New status to set.
        """
        async with self._lock:
            old = self._status.get(provider_name)
            self._status[provider_name] = status
            if old != status:
                logger.info(
                    "provider_status_changed",
                    provider=provider_name,
                    old_status=old.value if old else "unknown",
                    new_status=status.value,
                )

    async def _check_provider(self, name: str, provider: Any) -> None:
        """Run a single health check for one provider.

        Args:
            name: Provider identifier.
            provider: LLMProvider instance.
        """
        try:
            healthy = await provider.is_healthy()
            async with self._lock:
                if healthy:
                    self._status[name] = ProviderStatus.HEALTHY
                    self._failure_counts[name] = 0
                else:
                    self._failure_counts[name] += 1
                    fc = self._failure_counts[name]
                    if fc >= 3:
                        self._status[name] = ProviderStatus.UNAVAILABLE
                    else:
                        self._status[name] = ProviderStatus.DEGRADED

        except Exception as exc:
            async with self._lock:
                self._failure_counts[name] += 1
                fc = self._failure_counts[name]
                if fc >= 3:
                    self._status[name] = ProviderStatus.UNAVAILABLE
                else:
                    self._status[name] = ProviderStatus.DEGRADED

            logger.warning(
                "health_check_exception",
                provider=name,
                error=str(exc),
                failure_count=self._failure_counts.get(name, 0),
            )

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until cancelled."""
        while True:
            tasks = [
                self._check_provider(name, provider) for name, provider in self._providers.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate backoff based on worst-case failure count
            max_failures = max(self._failure_counts.values(), default=0)
            backoff_idx = min(max_failures, len(_BACKOFF_INTERVALS) - 1)
            interval = self._poll_interval if max_failures == 0 else _BACKOFF_INTERVALS[backoff_idx]

            await asyncio.sleep(interval)

    def start(self) -> None:
        """Start the background health monitoring task."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._poll_loop(), name="orion-health-monitor")
            logger.info(
                "health_monitor_started",
                providers=list(self._providers.keys()),
                poll_interval=self._poll_interval,
            )

    async def stop(self) -> None:
        """Stop the background health monitoring task."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            logger.info("health_monitor_stopped")

    async def run_initial_check(self) -> None:
        """Run one health check cycle immediately (for startup).

        Useful to populate initial status before accepting requests.
        """
        tasks = [self._check_provider(name, provider) for name, provider in self._providers.items()]
        await asyncio.gather(*tasks, return_exceptions=True)
