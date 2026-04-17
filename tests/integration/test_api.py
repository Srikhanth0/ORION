"""Integration tests for the ORION FastAPI REST API.

Uses httpx TestClient against the FastAPI app. All agent
pipeline calls are mocked to avoid real LLM calls.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from orion.api.server import app


@pytest.fixture()
def client() -> TestClient:
    """Create a clean test client."""
    # Reset task stores between tests
    from orion.api.routes import tasks

    tasks._tasks.clear()
    tasks._task_queues.clear()
    tasks._task_handles.clear()
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for /health and /ready."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /health always returns 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"  # type: ignore

    def test_ready_returns_status(self, client: TestClient) -> None:
        """GET /ready returns dependency checks."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "checks" in data


class TestTaskEndpoints:
    """Tests for /v1/tasks CRUD."""

    def test_submit_task_returns_202(self, client: TestClient) -> None:
        """POST /v1/tasks returns 202 with task_id."""
        resp = client.post(
            "/v1/tasks",
            json={"instruction": "Create a file"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "QUEUED"  # type: ignore

    def test_submit_task_validation_error(self, client: TestClient) -> None:
        """POST /v1/tasks with empty instruction returns 422."""
        resp = client.post(
            "/v1/tasks",
            json={"instruction": ""},
        )
        assert resp.status_code == 422

    def test_get_task_not_found(self, client: TestClient) -> None:
        """GET /v1/tasks/{id} with unknown ID returns 404."""
        resp = client.get("/v1/tasks/nonexistent")
        assert resp.status_code == 404

    def test_get_task_after_submit(self, client: TestClient) -> None:
        """GET /v1/tasks/{id} returns task after submission."""
        submit = client.post(
            "/v1/tasks",
            json={"instruction": "Test task"},
        )
        task_id = submit.json()["task_id"]

        resp = client.get(f"/v1/tasks/{task_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == task_id

    def test_cancel_task(self, client: TestClient) -> None:
        """DELETE /v1/tasks/{id} cancels a running task."""
        submit = client.post(
            "/v1/tasks",
            json={"instruction": "Long task"},
        )
        task_id = submit.json()["task_id"]

        resp = client.delete(f"/v1/tasks/{task_id}")
        assert resp.status_code == 204

    def test_cancel_nonexistent(self, client: TestClient) -> None:
        """DELETE /v1/tasks/{id} with unknown ID returns 404."""
        resp = client.delete("/v1/tasks/ghost")
        assert resp.status_code == 404

    def test_submit_with_context(self, client: TestClient) -> None:
        """POST /v1/tasks with context dict is accepted."""
        resp = client.post(
            "/v1/tasks",
            json={
                "instruction": "Do something",
                "context": {"os": "linux"},
                "timeout_seconds": 60,
            },
        )
        assert resp.status_code == 202


class TestToolEndpoints:
    """Tests for /v1/tools."""

    def test_list_tools(self, client: TestClient) -> None:
        """GET /v1/tools returns a list."""
        resp = client.get("/v1/tools")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_tool_not_found(self, client: TestClient) -> None:
        """GET /v1/tools/{name} with unknown name returns 404."""
        resp = client.get("/v1/tools/NONEXISTENT_TOOL")
        assert resp.status_code == 404
