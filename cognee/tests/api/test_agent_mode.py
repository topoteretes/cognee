import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["COGNEE_AGENT_MODE"] = "true"

    from fastapi.testclient import TestClient

from cognee.api.v1.agents import agent_mode

RUN_ID = uuid.uuid4().hex[:8]
OWNER_EMAIL = f"agentmode-owner-{RUN_ID}@example.com"
OWNER_PASSWORD = "ownerpass123!"


def _reset_agent_mode():
    """Reset module-level globals between tests."""
    agent_mode._active_count = 0
    agent_mode._watchdog_started = False


class TestAgentMode:
    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as client:
            yield client

    @pytest.fixture(scope="class")
    def owner_token(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert reg.status_code in (200, 201)

        login = client.post(
            "/api/v1/auth/login",
            data={"username": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert login.status_code == 200
        return login.json()["access_token"]

    @pytest.fixture(autouse=True)
    def reset_state(self):
        _reset_agent_mode()
        yield
        _reset_agent_mode()

    def test_register_increments_count(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        resp = client.post("/api/v1/agents/register", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 1

        resp = client.post("/api/v1/agents/register", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 2

    def test_unregister_decrements_count(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        client.post("/api/v1/agents/register", headers=headers)
        client.post("/api/v1/agents/register", headers=headers)

        resp = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 1

    def test_unregister_does_not_go_below_zero(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        resp = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 0

        resp = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp.json()["activeAgents"] == 0

    @patch.object(agent_mode, "_shutdown_server")
    def test_watchdog_does_not_shutdown_with_active_agents(self, mock_shutdown):
        agent_mode.register_agent()
        agent_mode._watchdog()
        mock_shutdown.assert_not_called()

    @patch.object(agent_mode, "_shutdown_server")
    def test_watchdog_shuts_down_when_no_agents(self, mock_shutdown):
        agent_mode._watchdog()
        mock_shutdown.assert_called_once()

    @patch.object(agent_mode, "_shutdown_server")
    def test_watchdog_shuts_down_after_all_unregister(self, mock_shutdown):
        agent_mode.register_agent()
        agent_mode.register_agent()
        agent_mode.unregister_agent()
        agent_mode.unregister_agent()

        agent_mode._watchdog()
        mock_shutdown.assert_called_once()

    def test_endpoints_reject_when_agent_mode_disabled(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        with patch.dict(os.environ, {"COGNEE_AGENT_MODE": "false"}):
            resp = client.post("/api/v1/agents/register", headers=headers)
            assert resp.status_code == 400
            assert "COGNEE_AGENT_MODE" in resp.json()["detail"]

            resp = client.post("/api/v1/agents/unregister", headers=headers)
            assert resp.status_code == 400
