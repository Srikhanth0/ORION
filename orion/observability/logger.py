"""Logger — structured logging configuration for ORION.

Configures structlog with:
- Dev: pretty console with colours.
- Prod: JSON to stdout.
- SensitiveFilter processor that redacts secrets.

Module Contract
---------------
- Call ``configure_logging()`` once at startup.
- All modules use ``structlog.get_logger(__name__)``.

Depends On
----------
- ``structlog``
"""
from __future__ import annotations

import logging
import os
import re
import sys

import structlog

# Patterns for sensitive value redaction
_SECRET_PATTERNS = re.compile(
    r"(api[_-]?key|secret|token|password|authorization"
    r"|bearer|credentials)"
    r"\s*[:=]\s*\S+",
    re.IGNORECASE,
)


class SensitiveFilter:
    """structlog processor that redacts sensitive values.

    Scans all string values in the event dict for patterns
    matching API keys, tokens, passwords, etc. Replaces
    matched values with ``[REDACTED]``.
    """

    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: dict,
    ) -> dict:
        """Process event dict and redact secrets.

        Args:
            logger: Logger instance.
            method_name: Log method name.
            event_dict: Event dictionary.

        Returns:
            Sanitised event dict.
        """
        for key, value in event_dict.items():
            if isinstance(value, str):
                event_dict[key] = _SECRET_PATTERNS.sub(
                    r"\1: [REDACTED]", value
                )
        return event_dict


class TaskContextFilter:
    """Injects task_id and agent into every log event.

    Uses thread-local or context-var storage for the
    current task context.
    """

    _context: dict[str, str] = {}

    @classmethod
    def set_context(
        cls,
        task_id: str = "",
        agent: str = "",
    ) -> None:
        """Set the current task context.

        Args:
            task_id: Current task ID.
            agent: Current agent name.
        """
        cls._context = {
            "task_id": task_id,
            "agent": agent,
        }

    @classmethod
    def clear_context(cls) -> None:
        """Clear the task context."""
        cls._context = {}

    def __call__(
        self,
        logger: logging.Logger,
        method_name: str,
        event_dict: dict,
    ) -> dict:
        """Inject task context into event dict.

        Args:
            logger: Logger instance.
            method_name: Log method name.
            event_dict: Event dictionary.

        Returns:
            Enriched event dict.
        """
        for key, val in self._context.items():
            if key not in event_dict and val:
                event_dict[key] = val
        return event_dict


def configure_logging(
    level: str = "INFO",
    force_json: bool = False,
) -> None:
    """Configure structlog for ORION.

    Dev mode (default): pretty console with colours.
    Prod mode (ORION_ENV=production or force_json): JSON.

    Args:
        level: Log level string.
        force_json: Force JSON output regardless of env.
    """
    is_prod = (
        force_json
        or os.environ.get("ORION_ENV") == "production"
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(
            fmt="iso", utc=True
        ),
        TaskContextFilter(),
        SensitiveFilter(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ]

    if is_prod:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(
                colors=True,
            ),
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
