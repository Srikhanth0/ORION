"""Robust JSON extraction and parsing utilities."""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def extract_json_array(text: str) -> list[Any] | None:
    """Extract a JSON array from model output, tolerating prose/fences.

    Ported from PlannerAgent for centralized reuse.
    """
    if not text:
        return None
    text = text.strip()

    # Strip markdown blocks
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        # Try to fix common LLM mistakes (like trailing commas before ])
        try:
            fixed = re.sub(r",\s*\]", "]", candidate)
            return json.loads(fixed)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            return None


def parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling code fences.

    Ported from BaseOrionAgent for centralized reuse.

    Args:
        text: Raw LLM output that should contain JSON.

    Returns:
        Parsed dict.

    Raises:
        ValueError: If JSON parsing fails.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        logger.warning(
            "json_parse_failed",
            error=str(exc),
            text_preview=cleaned[:200],
        )
        raise ValueError(f"Failed to parse JSON from LLM output: {exc}") from exc
