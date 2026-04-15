"""PermissionManifest — YAML-driven permission checking.

Loads rules from configs/safety/permissions.yaml and validates
tool invocations against category-specific allow/deny lists.
Handles shell command pattern matching and filesystem path
traversal prevention.

Module Contract
---------------
- **Inputs**: tool_name + params + category.
- **Outputs**: None (pass) or PermissionDeniedError (raise).

Depends On
----------
- ``orion.core.exceptions`` (PermissionDeniedError)
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import structlog

from orion.core.exceptions import PermissionDeniedError

logger = structlog.get_logger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


class PermissionManifest:
    """YAML-driven permission checker for tool invocations.

    Loads per-category rules from permissions.yaml:
    - ``allowed`` / ``denied``: exact action name lists
    - ``allowed_patterns`` / ``denied_patterns``: glob/regex for shell
    - ``allowed_paths`` / ``denied_paths``: filesystem boundaries

    Args:
        config_path: Path to permissions.yaml (auto-detected if None).
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
    ) -> None:
        self._rules: dict[str, Any] = {}
        path = config_path or (_CONFIGS_DIR / "safety" / "permissions.yaml")
        self._load(Path(path))

    def _load(self, path: Path) -> None:
        """Load permissions from YAML file.

        Args:
            path: Path to the YAML config.
        """
        if not path.exists():
            logger.warning(
                "permissions_yaml_missing",
                path=str(path),
            )
            return

        import yaml

        with open(path) as f:
            self._rules = yaml.safe_load(f) or {}

        logger.info(
            "permissions_loaded",
            categories=list(self._rules.keys()),
        )

    def check(
        self,
        tool_name: str,
        params: dict[str, Any],
        category: str | None = None,
    ) -> None:
        """Check if a tool invocation is allowed.

        Args:
            tool_name: Tool action name.
            params: Tool parameters.
            category: Tool category (github, shell, etc).

        Raises:
            PermissionDeniedError: If the invocation is denied.
        """
        if not self._rules:
            return  # No rules loaded — permit all

        cat = category or self._infer_category(tool_name)

        rules = self._rules.get(cat, {})
        if not rules:
            return  # No rules for this category

        # Check explicit denial first
        denied = rules.get("denied", [])
        if self._action_matches(tool_name, denied):
            logger.warning(
                "permission_denied",
                tool=tool_name,
                category=cat,
                rule="explicit_deny",
            )
            raise PermissionDeniedError(
                f"Tool '{tool_name}' is explicitly denied in category '{cat}'",
                tool_name=tool_name,
                action=tool_name,
                rule="explicit_deny",
            )

        # Category-specific checks
        if cat == "shell":
            self._check_shell(tool_name, params, rules)
        elif cat == "filesystem":
            self._check_filesystem(tool_name, params, rules)

        # Check allowed list if present
        allowed = rules.get("allowed", [])
        if allowed and not self._action_matches(tool_name, allowed):
            logger.warning(
                "permission_denied",
                tool=tool_name,
                category=cat,
                rule="not_in_allowed_list",
            )
            raise PermissionDeniedError(
                f"Tool '{tool_name}' is not in the allowed list for category '{cat}'",
                tool_name=tool_name,
                action=tool_name,
                rule="not_in_allowed_list",
            )

    def _check_shell(
        self,
        tool_name: str,
        params: dict[str, Any],
        rules: dict[str, Any],
    ) -> None:
        """Validate shell commands against patterns.

        Denied patterns are checked first. If any match, the command
        is denied. Then allowed patterns are checked — if none match,
        the command is denied.

        Args:
            tool_name: Tool name.
            params: Must contain 'command' key.
            rules: Shell category rules.

        Raises:
            PermissionDeniedError: If command is blocked.
        """
        command = params.get("command", "")
        if not command:
            return

        # Check denied patterns first
        denied_patterns = rules.get("denied_patterns", [])
        for pattern in denied_patterns:
            if pattern in command or re.search(re.escape(pattern), command):
                logger.warning(
                    "shell_command_denied",
                    command=command[:100],
                    pattern=pattern,
                )
                raise PermissionDeniedError(
                    f"Shell command matches denied pattern: '{pattern}'",
                    tool_name=tool_name,
                    action=command[:100],
                    rule=f"denied_pattern:{pattern}",
                )

        # Check allowed patterns
        allowed_patterns = rules.get("allowed_patterns", [])
        if allowed_patterns:
            matched = any(fnmatch.fnmatch(command, p) for p in allowed_patterns)
            if not matched:
                raise PermissionDeniedError(
                    "Shell command does not match any allowed pattern",
                    tool_name=tool_name,
                    action=command[:100],
                    rule="no_allowed_pattern_match",
                )

    def _check_filesystem(
        self,
        tool_name: str,
        params: dict[str, Any],
        rules: dict[str, Any],
    ) -> None:
        """Validate filesystem paths against allowed/denied lists.

        Resolves symlinks before comparison to prevent path
        traversal attacks.

        Args:
            tool_name: Tool name.
            params: Must contain 'path' key.
            rules: Filesystem category rules.

        Raises:
            PermissionDeniedError: If path is blocked.
        """
        raw_path = params.get("path", "")
        if not raw_path:
            return

        # Resolve symlinks to prevent traversal
        try:
            resolved = str(Path(raw_path).resolve())
        except (OSError, ValueError):
            resolved = raw_path

        # Check denied paths first
        denied_paths = rules.get("denied_paths", [])
        for dp in denied_paths:
            if resolved.startswith(dp):
                logger.warning(
                    "filesystem_path_denied",
                    path=resolved,
                    denied_prefix=dp,
                )
                raise PermissionDeniedError(
                    f"Path '{resolved}' is in denied area '{dp}'",
                    tool_name=tool_name,
                    action=resolved,
                    rule=f"denied_path:{dp}",
                )

        # Check allowed paths
        allowed_paths = rules.get("allowed_paths", [])
        if allowed_paths:
            in_allowed = any(resolved.startswith(ap) for ap in allowed_paths)
            if not in_allowed:
                raise PermissionDeniedError(
                    f"Path '{resolved}' is not in any allowed directory",
                    tool_name=tool_name,
                    action=resolved,
                    rule="not_in_allowed_path",
                )

    def _action_matches(self, tool_name: str, patterns: list[str]) -> bool:
        """Check if a tool name matches any pattern in a list.

        Args:
            tool_name: Tool action name.
            patterns: List of exact names or globs.

        Returns:
            True if the tool matches any pattern.
        """
        name_lower = tool_name.lower()
        for p in patterns:
            if p.lower() == name_lower:
                return True
            if fnmatch.fnmatch(name_lower, p.lower()):
                return True
        return False

    def _infer_category(self, tool_name: str) -> str:
        """Infer category from tool name.

        Args:
            tool_name: Tool action name.

        Returns:
            Category string.
        """
        upper = tool_name.upper()
        if upper.startswith("GITHUB"):
            return "github"
        if any(upper.startswith(p) for p in ("SHELL", "CMD", "OS")):
            return "shell"
        if upper.startswith(("FILE", "DIR")):
            return "filesystem"
        if upper.startswith("BROWSER"):
            return "browser"
        return "saas"
