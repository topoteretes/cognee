"""End-to-end tests for all /api/v1/agents/* endpoints.

Exercises every route in the merged agents router against a real FastAPI
TestClient with authentication enabled.  The operations layer is patched
only where it reaches into session-trace infrastructure that isn't available
in a lightweight test context — everything else (registry, models, DB) runs
for real.

Endpoints under test:
  POST   /register          — register agent connection
  GET    /                  — list agent connections
  GET    /{agent_id}        — agent connection detail
  POST   /create            — create sub-user agent
  GET    /list              — list agents
  DELETE /{agent_id}        — delete sub-user agent
  POST   /unregister        — agent-mode unregister
"""

import os
import uuid
import pytest
from unittest.mock import patch

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["HASH_API_KEY"] = "false"
    os.environ["COGNEE_AGENT_MODE"] = "true"

    from fastapi.testclient import TestClient

from cognee.api.v1.agents import agent_mode
from cognee.modules.agents.registry import AGENT_CONFIG_NAME, clear_registered_agent_connections

RUN_ID = uuid.uuid4().hex[:8]
OWNER_EMAIL = f"e2e-agents-{RUN_ID}@example.com"
OWNER_PASSWORD = "e2epass987!"
AGENT_NAME = f"e2e-agent-{RUN_ID}"


class TestAgentsE2E:
    """Full lifecycle test for every agents endpoint."""

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as c:
            yield c

    @pytest.fixture(scope="class")
    def owner(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert reg.status_code in (200, 201), reg.text
        owner_id = reg.json()["id"]

        login = client.post(
            "/api/v1/auth/login",
            data={"username": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert login.status_code == 200, login.text
        return {"id": owner_id, "token": login.json()["access_token"]}

    @pytest.fixture(scope="class")
    def headers(self, owner):
        return {"Authorization": f"Bearer {owner['token']}"}

    # ------------------------------------------------------------------ #
    # Agent connection endpoints
    # ------------------------------------------------------------------ #

    def test_list_connections_empty(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 0
        assert body["agents"] == []
        assert body["has_more"] is False

    def test_register_connection(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        resp = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "name": "support_bot",
                "type": "api",
                "memory_mode": "hybrid",
                "session_id": "sess-001",
                "dataset_ids": [],
                "dataset_names": ["company_brain"],
                "source": "api",
                "origin_function": "handle_ticket",
                "metadata": {"env": "test"},
            },
        )
        assert resp.status_code == 201, resp.text
        connection = resp.json()
        assert connection["name"] == "support_bot"
        assert connection["type"] == "api"
        assert connection["memory_mode"] == "hybrid"
        assert connection["session_id"] == "sess-001"
        assert connection["source"] == "api"
        assert connection["origin_function"] == "handle_ticket"
        assert connection["status"] == "active"
        assert connection["metadata"]["env"] == "test"
        assert len(connection["datasets"]) == 1
        assert connection["datasets"][0]["name"] == "company_brain"
        assert connection["datasets"][0]["type"] == "company_brain"
        assert connection["datasets"][0]["role"] == "read_write"

    def test_register_connection_defaults(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        resp = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={"name": "minimal_agent"},
        )
        assert resp.status_code == 201, resp.text
        connection = resp.json()
        assert connection["name"] == "minimal_agent"
        assert connection["type"] == "api"
        assert connection["memory_mode"] == "unknown"
        assert connection["source"] == "api"
        assert connection["datasets"] == []

    def test_list_connections_shows_registered(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        reg = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "name": "list_test_agent",
                "type": "sdk",
                "memory_mode": "cognee",
            },
        )
        assert reg.status_code == 201
        agent_id = reg.json()["id"]

        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["agents"][0]["id"] == agent_id
        assert body["agents"][0]["name"] == "list_test_agent"
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_connections_pagination(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        for i in range(3):
            client.post(
                "/api/v1/agents/register",
                headers=headers,
                json={"name": f"paginated_{i}"},
            )

        resp = client.get("/api/v1/agents/connections?limit=2&offset=0", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert len(body["agents"]) == 2
        assert body["has_more"] is True
        assert body["limit"] == 2
        assert body["offset"] == 0

        resp2 = client.get("/api/v1/agents/connections?limit=2&offset=2", headers=headers)
        body2 = resp2.json()
        assert len(body2["agents"]) == 1
        assert body2["has_more"] is False

    def test_list_connections_status_filter(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={"name": "active_agent"},
        )

        resp = client.get("/api/v1/agents/connections?status=active", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        resp2 = client.get("/api/v1/agents/connections?status=inactive", headers=headers)
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 0

    def test_get_connection_detail(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        reg = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "name": "detail_agent",
                "type": "mcp",
                "memory_mode": "session",
                "session_id": "detail-sess",
            },
        )
        assert reg.status_code == 201
        agent_id = reg.json()["id"]

        resp = client.get(f"/api/v1/agents/connections/{agent_id}", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"]["id"] == agent_id
        assert body["agent"]["name"] == "detail_agent"
        assert body["agent"]["type"] == "mcp"
        assert "memory_sources" in body
        assert "recent_sessions" in body
        assert "recent_traces" in body
        assert "recent_qas" in body

    def test_get_connection_detail_not_found(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        resp = client.get("/api/v1/agents/connections/nonexistent-id-12345", headers=headers)
        assert resp.status_code == 404
        assert resp.json()["detail"] == "connection not found"

    def test_register_multiple_connections_and_list(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        names = ["agent_alpha", "agent_beta", "agent_gamma"]
        ids = []
        for name in names:
            r = client.post(
                "/api/v1/agents/register",
                headers=headers,
                json={"name": name, "type": "sdk"},
            )
            assert r.status_code == 201
            ids.append(r.json()["id"])

        resp = client.get("/api/v1/agents/connections", headers=headers)
        body = resp.json()
        assert body["total"] == 3
        returned_ids = {a["id"] for a in body["agents"]}
        for agent_id in ids:
            assert agent_id in returned_ids

    def test_register_connection_with_dataset_ids(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        dataset_id = str(uuid.uuid4())
        resp = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "name": "dataset_agent",
                "dataset_ids": [dataset_id],
            },
        )
        assert resp.status_code == 201
        connection = resp.json()
        assert len(connection["datasets"]) == 1
        assert connection["datasets"][0]["id"] == dataset_id
        assert connection["datasets"][0]["role"] == "read_write"

    def test_register_connection_idempotent_update(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        payload = {
            "name": "idempotent_agent",
            "type": "api",
            "metadata": {"version": "1"},
        }
        r1 = client.post("/api/v1/agents/register", headers=headers, json=payload)
        assert r1.status_code == 201
        first_id = r1.json()["id"]

        payload["metadata"] = {"version": "2"}
        r2 = client.post("/api/v1/agents/register", headers=headers, json=payload)
        assert r2.status_code == 201
        assert r2.json()["id"] == first_id
        assert r2.json()["metadata"]["version"] == "2"

        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.json()["total"] == 1

    # ------------------------------------------------------------------ #
    # Sub-user agent CRUD endpoints
    # ------------------------------------------------------------------ #

    def test_create_sub_user_agent(self, client, headers):
        resp = client.post(
            f"/api/v1/agents/create?name={AGENT_NAME}",
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "agentId" in body
        assert "agentEmail" in body
        assert "agentApiKey" in body
        assert "@cognee.agent" in body["agentEmail"]

    def test_create_duplicate_sub_user_returns_409(self, client, headers):
        resp = client.post(
            f"/api/v1/agents/create?name={AGENT_NAME}",
            headers=headers,
        )
        assert resp.status_code == 409

    def test_list_agents(self, client, headers):
        resp = client.get("/api/v1/agents/list", headers=headers)
        assert resp.status_code == 200
        agents = resp.json()
        assert isinstance(agents, list)

    def test_sub_user_api_key_authenticates(self, client, headers):
        resp = client.post(
            f"/api/v1/agents/create?name=apikey-test-{RUN_ID}",
            headers=headers,
        )
        assert resp.status_code == 200
        api_key = resp.json()["agentApiKey"]

        me_resp = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": api_key},
        )
        assert me_resp.status_code == 200

    def test_delete_sub_user_agent(self, client, headers):
        create_resp = client.post(
            f"/api/v1/agents/create?name=to-delete-{RUN_ID}",
            headers=headers,
        )
        assert create_resp.status_code == 200
        agent_id = create_resp.json()["agentId"]

        del_resp = client.delete(
            f"/api/v1/agents/{agent_id}",
            headers=headers,
        )
        assert del_resp.status_code == 200

    def test_delete_nonexistent_agent_fails(self, client, headers):
        fake_id = str(uuid.uuid4())
        resp = client.delete(
            f"/api/v1/agents/{fake_id}",
            headers=headers,
        )
        assert resp.status_code in (403, 404, 500)

    # ------------------------------------------------------------------ #
    # Unregister endpoint
    # ------------------------------------------------------------------ #

    def test_unregister(self, client, headers):
        agent_mode._active_count = 2
        agent_mode._watchdog_started = False

        resp = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 1

        resp2 = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp2.json()["activeAgents"] == 0

    def test_unregister_floor_at_zero(self, client, headers):
        agent_mode._active_count = 0

        resp = client.post("/api/v1/agents/unregister", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 0

    # ------------------------------------------------------------------ #
    # Auth: unauthenticated requests rejected
    # ------------------------------------------------------------------ #

    def test_unauthenticated_rejected(self, client):
        client.post("/api/v1/auth/logout")
        saved_cookies = dict(client.cookies)
        client.cookies.clear()

        try:
            endpoints = [
                ("GET", "/api/v1/agents/connections"),
                ("POST", "/api/v1/agents/register"),
                ("GET", "/api/v1/agents/connections/some-id"),
                ("POST", "/api/v1/agents/create?name=nope"),
                ("GET", "/api/v1/agents/list"),
                ("DELETE", f"/api/v1/agents/{uuid.uuid4()}"),
                ("POST", "/api/v1/agents/unregister"),
            ]
            for method, path in endpoints:
                resp = client.request(method, path)
                assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}"
        finally:
            client.cookies.update(saved_cookies)

    # ------------------------------------------------------------------ #
    # Fixtures
    # ------------------------------------------------------------------ #

    @pytest.fixture(autouse=True)
    def _reset_agent_mode(self):
        saved = agent_mode._active_count
        yield
        agent_mode._active_count = saved

    @pytest.fixture(scope="class")
    def _patch_operations(self, owner):
        async def readable_datasets_for(_user):
            return []

        async def visible_user_ids(_user):
            return [uuid.UUID(owner["id"])]

        async def trace_agents_for_user(**_kwargs):
            return []

        async def persisted_agent_connections(_user_id):
            return []

        with (
            patch(
                "cognee.modules.agents.operations._readable_datasets_for",
                readable_datasets_for,
            ),
            patch(
                "cognee.modules.agents.operations._visible_user_ids",
                visible_user_ids,
            ),
            patch(
                "cognee.modules.agents.operations._trace_agents_for_user",
                trace_agents_for_user,
            ),
            patch(
                "cognee.modules.agents.operations.list_persisted_agent_connections",
                persisted_agent_connections,
            ),
        ):
            yield


PERSIST_RUN_ID = uuid.uuid4().hex[:8]
PERSIST_OWNER_EMAIL = f"persist-agents-{PERSIST_RUN_ID}@example.com"
PERSIST_OWNER_PASSWORD = "persistpass123!"
PERSIST_AGENT_NAME = f"persist-agent-{PERSIST_RUN_ID}"


class TestAgentPersistence:
    """Verify agent registration persists to principal_configuration
    without overwriting existing data."""

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as c:
            yield c

    @pytest.fixture(scope="class")
    def owner(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": PERSIST_OWNER_EMAIL, "password": PERSIST_OWNER_PASSWORD},
        )
        assert reg.status_code in (200, 201), reg.text
        owner_id = reg.json()["id"]

        login = client.post(
            "/api/v1/auth/login",
            data={"username": PERSIST_OWNER_EMAIL, "password": PERSIST_OWNER_PASSWORD},
        )
        assert login.status_code == 200, login.text
        return {"id": owner_id, "token": login.json()["access_token"]}

    @pytest.fixture(scope="class")
    def headers(self, owner):
        return {"Authorization": f"Bearer {owner['token']}"}

    @pytest.fixture(scope="class")
    def _patch_traces(self, owner):
        async def trace_agents_for_user(**_kwargs):
            return []

        with patch(
            "cognee.modules.agents.operations._trace_agents_for_user",
            trace_agents_for_user,
        ):
            yield

    def _seed_configuration(self, client, headers):
        resp = client.post(
            "/api/v1/configuration/store_user_configuration",
            json={"name": AGENT_CONFIG_NAME, "config": {"custom_setting": "keep_me", "agents": {}}},
            headers=headers,
        )
        assert resp.status_code == 200, resp.text

    def _read_agent_config(self, client, headers):
        resp = client.get(
            "/api/v1/configuration/get_user_configuration/",
            headers=headers,
        )
        assert resp.status_code == 200, resp.text
        for config in resp.json():
            if config.get("name") == AGENT_CONFIG_NAME:
                return config["configuration"]
        return None

    def test_register_persists_without_overwriting(self, client, headers, owner, _patch_traces):
        self._seed_configuration(client, headers)

        clear_registered_agent_connections()
        resp = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "name": PERSIST_AGENT_NAME,
                "type": "api",
                "memory_mode": "hybrid",
                "source": "api",
            },
        )
        assert resp.status_code == 201, resp.text
        connection = resp.json()
        agent_id = connection["id"]

        agent_config = self._read_agent_config(client, headers)

        assert agent_config is not None, "agent_configuration not found in principal_configuration"
        assert agent_config["custom_setting"] == "keep_me", "pre-existing config was overwritten"
        assert agent_id in agent_config["agents"], "registered agent not persisted"
        persisted = agent_config["agents"][agent_id]
        assert persisted["name"] == PERSIST_AGENT_NAME
        assert persisted["type"] == "api"
        assert persisted["memory_mode"] == "hybrid"

    def test_list_returns_persisted_agent(self, client, headers, owner, _patch_traces):
        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        agent_names = {a["name"] for a in body["agents"]}
        assert PERSIST_AGENT_NAME in agent_names

    def test_parent_sees_child_agent_connection(self, client, headers, owner, _patch_traces):
        child_name = f"child-agent-{PERSIST_RUN_ID}"
        create_resp = client.post(
            f"/api/v1/agents/create?name={child_name}",
            headers=headers,
        )
        assert create_resp.status_code == 200, create_resp.text
        child_api_key = create_resp.json()["agentApiKey"]

        child_headers = {"X-Api-Key": child_api_key}
        child_connection_name = f"child-bot-{PERSIST_RUN_ID}"
        reg_resp = client.post(
            "/api/v1/agents/register",
            headers=child_headers,
            json={
                "name": child_connection_name,
                "type": "sdk",
                "memory_mode": "cognee",
                "source": "api",
            },
        )
        assert reg_resp.status_code == 201, reg_resp.text
        child_connection_id = reg_resp.json()["id"]

        clear_registered_agent_connections()
        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        agent_ids = {a["id"] for a in body["agents"]}
        assert child_connection_id in agent_ids, (
            "parent user should see child agent's persisted connection"
        )
