"""Groq + OpenRouter quota tracking.

Thread-safe quota tracker that caches rate-limit information
from provider response headers. Updated after every API call.

Module Contract
---------------
- **Inputs**: Raw response headers from Groq/OpenRouter.
- **Outputs**: ``QuotaInfo`` snapshots.
- **Thread Safety**: Uses ``asyncio.Lock`` for concurrent access.

Depends On
----------
- ``orion.llm.providers.base`` (QuotaInfo)
"""

from __future__ import annotations

import asyncio
import contextlib
import time

from orion.llm.providers.base import QuotaInfo


class QuotaTracker:
    """Tracks rate-limit quota from provider response headers.

    Updated after every API call by parsing ``x-ratelimit-*`` headers.
    Used by the AdaptiveLLMRouter to pre-emptively skip exhausted
    or nearly-exhausted providers.

    Args:
        provider_name: The provider this tracker is for (e.g., 'groq').
        daily_budget_usd: Optional daily spend cap (for pay-per-token providers).
    """

    def __init__(
        self,
        provider_name: str,
        daily_budget_usd: float | None = None,
    ) -> None:
        self._provider = provider_name
        self._lock = asyncio.Lock()

        # Rate-limit fields (from response headers)
        self._remaining_requests: int | None = None
        self._remaining_tokens: int | None = None
        self._reset_at_epoch: float | None = None

        # Budget tracking (for OpenRouter)
        self._daily_budget_usd = daily_budget_usd
        self._daily_spend_usd: float = 0.0
        self._budget_reset_epoch: float = self._next_midnight()

    async def update_from_headers(self, headers: dict[str, str]) -> None:
        """Parse and cache rate-limit info from response headers.

        Handles both Groq-style and OpenRouter-style headers:
        - ``x-ratelimit-remaining-requests``
        - ``x-ratelimit-remaining-tokens``
        - ``x-ratelimit-reset-requests`` (ISO8601 or epoch)

        Args:
            headers: Raw HTTP response headers dict.
        """
        async with self._lock:
            # Groq/OpenRouter rate-limit headers
            if "x-ratelimit-remaining-requests" in headers:
                with contextlib.suppress(ValueError, TypeError):
                    self._remaining_requests = int(headers["x-ratelimit-remaining-requests"])

            if "x-ratelimit-remaining-tokens" in headers:
                with contextlib.suppress(ValueError, TypeError):
                    self._remaining_tokens = int(headers["x-ratelimit-remaining-tokens"])

            if "x-ratelimit-reset-requests" in headers:
                with contextlib.suppress(ValueError, TypeError):
                    self._reset_at_epoch = float(headers["x-ratelimit-reset-requests"])

    async def record_spend(self, cost_usd: float) -> None:
        """Record a spend amount for budget tracking.

        Args:
            cost_usd: Cost of the last request in USD.
        """
        async with self._lock:
            # Reset daily budget at midnight
            now = time.time()
            if now > self._budget_reset_epoch:
                self._daily_spend_usd = 0.0
                self._budget_reset_epoch = self._next_midnight()

            self._daily_spend_usd += cost_usd

    async def get_quota(self) -> QuotaInfo:
        """Return a snapshot of current quota status.

        Returns:
            QuotaInfo with all available quota metrics.
        """
        async with self._lock:
            daily_remaining: float | None = None
            if self._daily_budget_usd is not None:
                daily_remaining = self._daily_budget_usd - self._daily_spend_usd

            return QuotaInfo(
                remaining_requests=self._remaining_requests,
                remaining_tokens=self._remaining_tokens,
                daily_budget_remaining_usd=daily_remaining,
                reset_at_epoch=self._reset_at_epoch,
            )

    async def reset(self) -> None:
        """Reset all tracked quota values."""
        async with self._lock:
            self._remaining_requests = None
            self._remaining_tokens = None
            self._reset_at_epoch = None
            self._daily_spend_usd = 0.0

    @staticmethod
    def _next_midnight() -> float:
        """Return epoch timestamp of next UTC midnight."""
        import calendar
        from datetime import UTC, datetime, timedelta

        now = datetime.now(UTC)
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return float(calendar.timegm(tomorrow.timetuple()))
