"""Unit tests for the safety layer (PermissionManifest, RollbackEngine,
DestructiveOpGate, ExecSandbox)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from orion.core.exceptions import (
    PermissionDeniedError,
    SandboxViolationError,
)
from orion.safety.gate import DestructiveOpGate
from orion.safety.manifest import PermissionManifest
from orion.safety.rollback import RollbackEngine
from orion.safety.sandbox import ExecSandbox

# ── PermissionManifest Tests ───────────────────────────────


class TestPermissionManifest:
    """Tests for PermissionManifest."""

    @pytest.fixture()
    def manifest(self, tmp_path: Path) -> PermissionManifest:
        """Create a manifest with test rules."""
        config = tmp_path / "permissions.yaml"
        # Use tmp_path in allowed_paths so tests work on Windows
        tp = str(tmp_path).replace("\\", "\\\\")
        config.write_text(
            f"""
github:
  allowed:
    - create_issue
    - list_prs
  denied:
    - delete_repo
shell:
  denied_patterns:
    - "rm -rf /"
    - "sudo rm"
  allowed_patterns:
    - "*"
filesystem:
  allowed_paths:
    - "/tmp"
    - "/home"
    - "{tp}"
  denied_paths:
    - "/etc"
    - "/sys"
"""
        )
        return PermissionManifest(config_path=config)

    def test_allowed_action_passes(self, manifest: PermissionManifest) -> None:
        """Allowed GitHub action passes check."""
        manifest.check("create_issue", {}, category="github")

    def test_denied_action_raises(self, manifest: PermissionManifest) -> None:
        """Denied GitHub action raises PermissionDeniedError."""
        with pytest.raises(PermissionDeniedError, match="delete_repo"):
            manifest.check("delete_repo", {}, category="github")

    def test_shell_denied_pattern(self, manifest: PermissionManifest) -> None:
        """Shell command matching denied pattern is blocked."""
        with pytest.raises(PermissionDeniedError):
            manifest.check(
                "exec_cmd",
                {"command": "sudo rm -rf /"},
                category="shell",
            )

    def test_shell_safe_command(self, manifest: PermissionManifest) -> None:
        """Safe shell command passes."""
        manifest.check(
            "exec_cmd",
            {"command": "ls -la"},
            category="shell",
        )

    def test_filesystem_denied_path(self, manifest: PermissionManifest) -> None:
        """Filesystem path in denied area is blocked."""
        with pytest.raises(PermissionDeniedError):
            manifest.check(
                "FILE_READ",
                {"path": "/etc/passwd"},
                category="filesystem",
            )

    def test_filesystem_allowed_path(self, manifest: PermissionManifest, tmp_path: Path) -> None:
        """Filesystem path in allowed area passes."""
        test_path = str(tmp_path / "test.txt")
        manifest.check(
            "FILE_READ",
            {"path": test_path},
            category="filesystem",
        )

    def test_unknown_category_passes(self, manifest: PermissionManifest) -> None:
        """Tool with no matching category rules passes."""
        manifest.check("CUSTOM_TOOL", {}, category="custom")

    def test_not_in_allowed_list_denied(self, manifest: PermissionManifest) -> None:
        """GitHub action not in allowed list is denied."""
        with pytest.raises(PermissionDeniedError):
            manifest.check("transfer_repo", {}, category="github")


# ── RollbackEngine Tests ──────────────────────────────────


class TestRollbackEngine:
    """Tests for RollbackEngine."""

    @pytest.fixture()
    def engine(self, tmp_path: Path) -> RollbackEngine:
        """Create an engine with temp checkpoint dir."""
        return RollbackEngine(checkpoint_dir=tmp_path / "checkpoints")

    def test_checkpoint_creates_point(self, engine: RollbackEngine) -> None:
        """checkpoint() creates a RollbackPoint."""
        point = engine.checkpoint(
            "s1",
            "FILE_WRITE",
            {"path": "/tmp/test.txt"},
            task_id="t1",
        )
        assert point.subtask_id == "s1"
        assert point.tool_name == "FILE_WRITE"
        assert engine.has_checkpoints("t1")

    def test_rollback_file_restore(self, engine: RollbackEngine, tmp_path: Path) -> None:
        """Rollback restores original file content."""
        test_file = tmp_path / "rollback_test.txt"
        test_file.write_text("original")

        engine.checkpoint(
            "s1",
            "file_write",
            {"path": str(test_file)},
            task_id="t1",
        )

        # Simulate modification
        test_file.write_text("modified")
        assert test_file.read_text() == "modified"

        # Rollback
        results = engine.rollback("t1")
        assert any("[OK]" in r for r in results)
        assert test_file.read_text() == "original"

    def test_rollback_file_delete_if_new(self, engine: RollbackEngine, tmp_path: Path) -> None:
        """Rollback deletes a file that didn't exist before."""
        new_file = tmp_path / "new_file.txt"

        engine.checkpoint(
            "s1",
            "file_write",
            {"path": str(new_file)},
            task_id="t1",
        )

        # Simulate creation
        new_file.write_text("created")

        # Rollback
        engine.rollback("t1")
        assert not new_file.exists()

    def test_rollback_irreversible_warning(self, engine: RollbackEngine) -> None:
        """Irreversible ops produce a warning but don't block."""
        engine.checkpoint(
            "s1",
            "send_email",
            {"to": "test@example.com"},
            task_id="t1",
        )

        results = engine.rollback("t1")
        assert any("IRREVERSIBLE" in r for r in results)

    def test_rollback_lifo_order(self, engine: RollbackEngine, tmp_path: Path) -> None:
        """Checkpoints are rolled back in LIFO order."""
        f1 = tmp_path / "f1.txt"
        f2 = tmp_path / "f2.txt"

        engine.checkpoint(
            "s1",
            "file_write",
            {"path": str(f1)},
            task_id="t1",
        )
        engine.checkpoint(
            "s2",
            "file_write",
            {"path": str(f2)},
            task_id="t1",
        )

        # Simulate creation
        f1.write_text("f1")
        f2.write_text("f2")

        results = engine.rollback("t1")
        # s2 should be rolled back first (LIFO)
        assert "s2" in results[0]

    def test_no_checkpoints_rollback(self, engine: RollbackEngine) -> None:
        """Rollback with no checkpoints returns info message."""
        results = engine.rollback("nonexistent")
        assert "No checkpoints" in results[0]


# ── DestructiveOpGate Tests ───────────────────────────────


class TestDestructiveOpGate:
    """Tests for DestructiveOpGate."""

    @pytest.mark.asyncio
    async def test_auto_low_risk_approved(self) -> None:
        """AUTO mode: low-risk ops are auto-approved."""
        gate = DestructiveOpGate(mode="auto")
        result = await gate.approve("delete_file", {"path": "/tmp/cache.txt"})
        assert result.approved is True
        assert result.risk_level == "low"

    @pytest.mark.asyncio
    async def test_auto_high_risk_no_hitl_approved(
        self,
    ) -> None:
        """AUTO mode: high-risk ops without HITL auto-approve."""
        gate = DestructiveOpGate(mode="auto")
        result = await gate.approve("delete_repo", {"repo": "org/production"})
        assert result.approved is True  # auto-approve no hitl

    @pytest.mark.asyncio
    async def test_strict_no_hitl_denied(self) -> None:
        """STRICT mode: denied without HITL gateway."""
        gate = DestructiveOpGate(mode="strict")
        result = await gate.approve("delete_file", {"path": "/tmp/test"})
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_strict_with_hitl_approval(self) -> None:
        """STRICT mode: approved via HITL gateway."""

        class MockHITL:
            async def ask_human(self, ctx: Any) -> bool:
                return True

        gate = DestructiveOpGate(mode="strict", hitl_gateway=MockHITL())
        result = await gate.approve("delete_file", {})
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_risk_assessment_high(self) -> None:
        """High-risk patterns are correctly identified."""
        gate = DestructiveOpGate(mode="auto")
        risk = gate._assess_risk("exec_cmd", {"command": "rm -rf /"})
        assert risk == "high"

    @pytest.mark.asyncio
    async def test_risk_assessment_low(self) -> None:
        """Low-risk patterns are correctly identified."""
        gate = DestructiveOpGate(mode="auto")
        risk = gate._assess_risk("delete_file", {})
        assert risk == "low"


# ── ExecSandbox Tests ─────────────────────────────────────


class TestExecSandbox:
    """Tests for ExecSandbox."""

    def test_cwd_validation_denied(self) -> None:
        """CWD outside allowed paths raises SandboxViolationError."""
        sandbox = ExecSandbox(allowed_paths=["/tmp"])
        with pytest.raises(SandboxViolationError):
            sandbox._validate_cwd("/etc/dangerous")

    def test_cwd_validation_allowed(self) -> None:
        """CWD inside allowed paths passes."""
        sandbox = ExecSandbox(allowed_paths=["/tmp", str(Path.cwd())])
        sandbox._validate_cwd(str(Path.cwd()))

    def test_env_sanitization(self) -> None:
        """Sensitive env vars are stripped."""
        import os

        os.environ["TEST_API_KEY_SECRET"] = "sensitive"
        sandbox = ExecSandbox()
        env = sandbox._sanitize_env()

        assert "TEST_API_KEY_SECRET" not in env

        del os.environ["TEST_API_KEY_SECRET"]
