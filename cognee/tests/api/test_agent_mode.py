import os
import uuid
import pytest
from unittest.mock import patch, MagicMock

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["COGNEE_AGENT_MODE"] = "true"

    from fastapi.testclient import TestClient

from cognee.modules.agents import agent_mode
from cognee.modules.agents.models import RegisterAgentRequest, UnregisterAgentRequest
from cognee.modules.users.models.User import User

RUN_ID = uuid.uuid4().hex[:8]
OWNER_EMAIL = f"agentmode-owner-{RUN_ID}@example.com"
OWNER_PASSWORD = "ownerpass123!"

REGISTER_BODY = {"agent_session_name": "test-agent"}
REGISTER_BODY_2 = {"agent_session_name": "test-agent-2"}

_DUMMY_USER = User(email="test@test.com", hashed_password="!")
_DUMMY_USER_2 = User(email="test2@test.com", hashed_password="!")
_DUMMY_REQUEST = RegisterAgentRequest(agent_session_name="test-agent")
_DUMMY_REQUEST_2 = RegisterAgentRequest(agent_session_name="test-agent-2")


def _reset_agent_mode():
    """Reset module-level globals between tests."""
    from cognee.modules.agents.registry import clear_registered_agent_connections

    agent_mode._active_count = 0
    agent_mode._active_connection_ids.clear()
    agent_mode._watchdog_started = False
    clear_registered_agent_connections()


@pytest.fixture(autouse=True)
def _patch_persistence(monkeypatch):
    async def noop_persist(_user_id, _connection):
        pass

    async def noop_deactivate(_user_id, _connection_id):
        pass

    monkeypatch.setattr(
        "cognee.modules.agents.registry._persist_agent_connection",
        noop_persist,
    )
    monkeypatch.setattr(
        "cognee.modules.agents.registry._deactivate_persisted_connection",
        noop_deactivate,
    )


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

        resp = client.post("/api/v1/agents/register", json=REGISTER_BODY, headers=headers)
        assert resp.status_code == 201
        assert agent_mode._active_count == 1

    def test_two_connections_increment_separately(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        client.post("/api/v1/agents/register", json=REGISTER_BODY, headers=headers)
        client.post("/api/v1/agents/register", json=REGISTER_BODY_2, headers=headers)
        assert agent_mode._active_count == 2

    def test_unregister_specific_connection(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        client.post("/api/v1/agents/register", json=REGISTER_BODY, headers=headers)
        client.post("/api/v1/agents/register", json=REGISTER_BODY_2, headers=headers)
        assert agent_mode._active_count == 2

        resp = client.post(
            "/api/v1/agents/unregister",
            json={"agent_session_name": "test-agent"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 1

    def test_unregister_does_not_go_below_zero(self, client, owner_token):
        headers = {"Authorization": f"Bearer {owner_token}"}

        resp = client.post(
            "/api/v1/agents/unregister",
            json={"agent_session_name": "test-agent"},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 0

    @pytest.mark.asyncio
    @patch.object(agent_mode, "_shutdown_server")
    async def test_watchdog_does_not_shutdown_with_active_agents(self, mock_shutdown):
        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
        agent_mode._watchdog()
        mock_shutdown.assert_not_called()

    @patch.object(agent_mode, "_shutdown_server")
    def test_watchdog_shuts_down_when_no_agents(self, mock_shutdown):
        agent_mode._watchdog()
        mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(agent_mode, "_shutdown_server")
    async def test_watchdog_shuts_down_after_all_unregister(self, mock_shutdown):
        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
        await agent_mode.register_agent(_DUMMY_USER_2, _DUMMY_REQUEST_2)
        await agent_mode.unregister_agent(
            _DUMMY_USER, UnregisterAgentRequest(agent_session_name="test-agent")
        )
        await agent_mode.unregister_agent(
            _DUMMY_USER_2, UnregisterAgentRequest(agent_session_name="test-agent-2")
        )

        agent_mode._watchdog()
        mock_shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_watchdog_does_not_start_when_agent_mode_disabled(self):
        with patch.dict(os.environ, {"COGNEE_AGENT_MODE": "false"}):
            await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
            assert not agent_mode._watchdog_started

    @pytest.mark.asyncio
    async def test_re_register_same_connection_does_not_increment(self):
        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
        assert agent_mode._active_count == 1

        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
        assert agent_mode._active_count == 1

    @pytest.mark.asyncio
    async def test_same_user_different_connections_increment(self):
        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST)
        await agent_mode.register_agent(_DUMMY_USER, _DUMMY_REQUEST_2)
        assert agent_mode._active_count == 2
