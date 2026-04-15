"""DestructiveOpGate — approval gate for destructive operations.

Routes destructive tool invocations through a human-in-the-loop
approval flow. Operates in AUTO or STRICT mode based on config.

Module Contract
---------------
- **Inputs**: tool_name + params + rollback_available.
- **Outputs**: ApprovalResult(approved, reason).

Depends On
----------
- ``structlog`` (logging)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Risk levels for destructive operations
_HIGH_RISK_PATTERNS = frozenset(
    {
        "rm -rf",
        "drop",
        "delete_repo",
        "truncate",
        "format",
        "destroy",
        "bulk_delete",
    }
)

_LOW_RISK_PATTERNS = frozenset(
    {
        "delete_file",
        "remove_temp",
        "clean_cache",
        "unlink",
    }
)


@dataclass
class ApprovalResult:
    """Result of a destructive operation approval check.

    Attributes:
        approved: Whether the operation is approved.
        reason: Human-readable reason for the decision.
        risk_level: Assessed risk (low, high).
        mode: Gate mode used (auto, strict).
    """

    approved: bool
    reason: str
    risk_level: str = "unknown"
    mode: str = "auto"


class DestructiveOpGate:
    """Approval gate for destructive tool operations.

    Modes:
    - AUTO: Low-risk ops auto-approve with warning.
      High-risk ops require human approval.
    - STRICT: ALL destructive ops require human approval.

    Args:
        mode: Gate mode ('auto' or 'strict').
        hitl_gateway: HITL gateway for human approval.
        destructive_ops: List of operations classified as destructive.
    """

    def __init__(
        self,
        mode: str = "auto",
        hitl_gateway: Any = None,
        destructive_ops: list[str] | None = None,
    ) -> None:
        self._mode = mode.lower()
        self._hitl = hitl_gateway
        self._destructive_ops = set(destructive_ops or [])

    async def approve(
        self,
        tool_name: str,
        params: dict[str, Any],
        rollback_available: bool = False,
    ) -> ApprovalResult:
        """Check if a destructive operation is approved.

        Args:
            tool_name: Tool action name.
            params: Tool parameters.
            rollback_available: Whether rollback is possible.

        Returns:
            ApprovalResult with decision.
        """
        risk_level = self._assess_risk(tool_name, params)

        # STRICT mode: always require approval
        if self._mode == "strict":
            logger.info(
                "gate_strict_approval_required",
                tool=tool_name,
                risk=risk_level,
            )
            return await self._request_human_approval(
                tool_name,
                params,
                risk_level,
                rollback_available,
            )

        # AUTO mode
        if risk_level == "high":
            logger.info(
                "gate_high_risk_approval_required",
                tool=tool_name,
            )
            return await self._request_human_approval(
                tool_name,
                params,
                risk_level,
                rollback_available,
            )

        # Low risk: auto-approve with warning
        logger.warning(
            "gate_auto_approved",
            tool=tool_name,
            risk=risk_level,
        )
        return ApprovalResult(
            approved=True,
            reason=(f"Auto-approved low-risk destructive op: {tool_name}"),
            risk_level=risk_level,
            mode=self._mode,
        )

    def _assess_risk(
        self,
        tool_name: str,
        params: dict[str, Any],
    ) -> str:
        """Assess the risk level of a destructive operation.

        Args:
            tool_name: Tool name.
            params: Tool parameters.

        Returns:
            Risk level string ('low' or 'high').
        """
        name_lower = tool_name.lower()
        command = str(params.get("command", "")).lower()
        combined = f"{name_lower} {command}"

        if any(p in combined for p in _HIGH_RISK_PATTERNS):
            return "high"

        if any(p in combined for p in _LOW_RISK_PATTERNS):
            return "low"

        # Default: high for unknown destructive ops
        return "high"

    async def _request_human_approval(
        self,
        tool_name: str,
        params: dict[str, Any],
        risk_level: str,
        rollback_available: bool,
    ) -> ApprovalResult:
        """Request human approval via HITL gateway.

        Falls back to auto-deny if no gateway is configured.

        Args:
            tool_name: Tool name.
            params: Tool parameters.
            risk_level: Assessed risk level.
            rollback_available: Whether rollback is possible.

        Returns:
            ApprovalResult from human or default deny.
        """
        if self._hitl is None:
            # No HITL gateway — deny by default in strict
            if self._mode == "strict":
                return ApprovalResult(
                    approved=False,
                    reason=("No HITL gateway configured for strict mode approval"),
                    risk_level=risk_level,
                    mode=self._mode,
                )
            # Auto mode, no HITL — auto-approve
            return ApprovalResult(
                approved=True,
                reason="No HITL gateway; auto-approved",
                risk_level=risk_level,
                mode=self._mode,
            )

        # Use HITL gateway
        try:
            if hasattr(self._hitl, "ask_human"):
                context = {
                    "tool": tool_name,
                    "params": params,
                    "risk_level": risk_level,
                    "rollback_available": rollback_available,
                }
                response = await self._hitl.ask_human(context)
                return ApprovalResult(
                    approved=bool(response),
                    reason=str(response),
                    risk_level=risk_level,
                    mode=self._mode,
                )
        except Exception as exc:
            logger.error(
                "gate_hitl_failed",
                tool=tool_name,
                error=str(exc),
            )

        return ApprovalResult(
            approved=False,
            reason="HITL gateway error — denied for safety",
            risk_level=risk_level,
            mode=self._mode,
        )
