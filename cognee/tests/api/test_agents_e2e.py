"""End-to-end tests for all /api/v1/agents/* endpoints.

Exercises every route in the merged agents router against a real FastAPI
TestClient with authentication enabled.  The operations layer is patched
only where it reaches into session-trace infrastructure that isn't available
in a lightweight test context — everything else (registry, models, DB) runs
for real.

Endpoints under test:
  GET    /connections            — list agent connections
  GET    /connections/{agent_id} — agent connection detail
  POST   /register              — register agent connection
  POST   /unregister            — unregister agent connection
  GET    /list                  — list agents from DB
  POST   /create                — create agent
  GET    /{agent_id}            — get agent from DB
  DELETE /{agent_id}            — delete agent
"""

import os
import uuid
from types import SimpleNamespace
import pytest
from unittest.mock import MagicMock, patch

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["HASH_API_KEY"] = "false"
    os.environ["COGNEE_AGENT_MODE"] = "true"

    from fastapi.testclient import TestClient

from cognee.modules.agents import agent_mode
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
                "agent_session_name": "support_bot",
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
        assert connection["agent_session_name"] == "support_bot"
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
            json={"agent_session_name": "minimal_agent"},
        )
        assert resp.status_code == 201, resp.text
        connection = resp.json()
        assert connection["agent_session_name"] == "minimal_agent"
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
                "agent_session_name": "list_test_agent",
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
        assert body["agents"][0]["agent_session_name"] == "list_test_agent"
        assert body["limit"] == 50
        assert body["offset"] == 0

    def test_list_connections_pagination(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        for i in range(3):
            client.post(
                "/api/v1/agents/register",
                headers=headers,
                json={"agent_session_name": f"paginated_{i}"},
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
            json={"agent_session_name": "active_agent"},
        )

        resp = client.get("/api/v1/agents/connections?status=active", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["total"] == 1

        resp2 = client.get("/api/v1/agents/connections?status=inactive", headers=headers)
        assert resp2.status_code == 200
        assert resp2.json()["total"] == 0

    def test_get_connection_detail(self, client, headers, owner, _patch_operations):
        clear_registered_agent_connections()
        reg = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "agent_session_name": "detail_agent",
                "type": "mcp",
                "memory_mode": "session",
                "session_id": "detail-sess",
            },
        )
        assert reg.status_code == 201

        resp = client.get(
            f"/api/v1/agents/connections/{owner['id']}?agent_session_name=detail_agent",
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["agent"]["agent_session_name"] == "detail_agent"
        assert body["agent"]["type"] == "mcp"
        assert "memory_sources" in body
        assert "recent_sessions" in body
        assert "recent_traces" in body
        assert "recent_qas" in body

    def test_get_connection_detail_not_found(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        fake_id = str(uuid.uuid4())
        resp = client.get(f"/api/v1/agents/connections/{fake_id}", headers=headers)
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
                json={"agent_session_name": name, "type": "sdk"},
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
                "agent_session_name": "dataset_agent",
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
            "agent_session_name": "idempotent_agent",
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

    def test_unregister(self, client, headers, _patch_operations):
        clear_registered_agent_connections()
        agent_mode._active_count = 0
        agent_mode._active_connection_ids.clear()

        client.post(
            "/api/v1/agents/register", headers=headers, json={"agent_session_name": "unreg-a"}
        )
        client.post(
            "/api/v1/agents/register", headers=headers, json={"agent_session_name": "unreg-b"}
        )
        assert agent_mode._active_count == 2

        resp = client.post(
            "/api/v1/agents/unregister", json={"agent_session_name": "unreg-a"}, headers=headers
        )
        assert resp.status_code == 200
        assert resp.json()["activeAgents"] == 1

        resp2 = client.post(
            "/api/v1/agents/unregister", json={"agent_session_name": "unreg-b"}, headers=headers
        )
        assert resp2.json()["activeAgents"] == 0

    def test_unregister_floor_at_zero(self, client, headers):
        agent_mode._active_count = 0
        agent_mode._active_connection_ids.clear()

        resp = client.post(
            "/api/v1/agents/unregister", json={"agent_session_name": "nonexistent"}, headers=headers
        )
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
                ("GET", "/api/v1/agents/connections", None),
                ("POST", "/api/v1/agents/register", {"agent_session_name": "x"}),
                ("GET", f"/api/v1/agents/connections/{uuid.uuid4()}", None),
                ("POST", "/api/v1/agents/create?name=nope", None),
                ("GET", "/api/v1/agents/list", None),
                ("DELETE", f"/api/v1/agents/{uuid.uuid4()}", None),
                ("POST", "/api/v1/agents/unregister", {"agent_session_name": "x"}),
            ]
            for method, path, body in endpoints:
                resp = client.request(method, path, json=body)
                assert resp.status_code == 401, f"{method} {path} returned {resp.status_code}"
        finally:
            client.cookies.update(saved_cookies)

    # ------------------------------------------------------------------ #
    # Fixtures
    # ------------------------------------------------------------------ #

    @pytest.fixture(autouse=True)
    def _reset_agent_mode(self):
        saved_count = agent_mode._active_count
        saved_ids = set(agent_mode._active_connection_ids)
        yield
        agent_mode._active_count = saved_count
        agent_mode._active_connection_ids.clear()
        agent_mode._active_connection_ids.update(saved_ids)

    @pytest.fixture(scope="class")
    def _patch_operations(self, owner):
        async def readable_datasets_for(_user):
            return []

        async def visible_user_ids(_user):
            return [uuid.UUID(owner["id"])]

        async def persisted_agent_connections(_user_id, active_only=True):
            return []

        async def authorized_dataset(_user, dataset_id, _permission):
            return SimpleNamespace(id=dataset_id, name="project_dataset")

        async def authorized_dataset_by_name(dataset_name, _user, _permission):
            return SimpleNamespace(
                id=uuid.uuid5(uuid.NAMESPACE_URL, dataset_name), name=dataset_name
            )

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
                "cognee.modules.agents.operations.list_persisted_agent_connections",
                persisted_agent_connections,
            ),
            patch(
                "cognee.modules.agents.operations.get_authorized_dataset",
                authorized_dataset,
            ),
            patch(
                "cognee.modules.agents.operations.get_authorized_dataset_by_name",
                authorized_dataset_by_name,
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

    def test_register_persists_without_overwriting(self, client, headers, owner):
        self._seed_configuration(client, headers)

        clear_registered_agent_connections()
        resp = client.post(
            "/api/v1/agents/register",
            headers=headers,
            json={
                "agent_session_name": PERSIST_AGENT_NAME,
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
        assert persisted["agent_session_name"] == PERSIST_AGENT_NAME
        assert persisted["type"] == "api"
        assert persisted["memory_mode"] == "hybrid"

    def test_list_returns_persisted_agent(self, client, headers, owner):
        resp = client.get("/api/v1/agents/connections", headers=headers)
        assert resp.status_code == 200
        body = resp.json()
        agent_names = {a["agent_session_name"] for a in body["agents"]}
        assert PERSIST_AGENT_NAME in agent_names

    def test_parent_sees_child_agent_connection(self, client, headers, owner):
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
                "agent_session_name": child_connection_name,
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


LIFECYCLE_RUN_ID = uuid.uuid4().hex[:8]
LIFECYCLE_OWNER_EMAIL = f"lifecycle-{LIFECYCLE_RUN_ID}@example.com"
LIFECYCLE_OWNER_PASSWORD = "lifecycle123!"


class TestAgentFullLifecycle:
    """Full lifecycle: create agents, register them, verify connections,
    unregister, confirm cleanup, and test watchdog shutdown."""

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as c:
            yield c

    @pytest.fixture(scope="class")
    def owner(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": LIFECYCLE_OWNER_EMAIL, "password": LIFECYCLE_OWNER_PASSWORD},
        )
        assert reg.status_code in (200, 201), reg.text
        owner_id = reg.json()["id"]

        login = client.post(
            "/api/v1/auth/login",
            data={"username": LIFECYCLE_OWNER_EMAIL, "password": LIFECYCLE_OWNER_PASSWORD},
        )
        assert login.status_code == 200, login.text
        return {"id": owner_id, "token": login.json()["access_token"]}

    @pytest.fixture(scope="class")
    def headers(self, owner):
        return {"Authorization": f"Bearer {owner['token']}"}

    @pytest.fixture(autouse=True)
    def _reset(self):
        agent_mode._active_count = 0
        agent_mode._active_connection_ids.clear()
        agent_mode._watchdog_started = False
        clear_registered_agent_connections()
        yield

    def test_full_lifecycle(self, client, headers, owner):
        agent_a_name = f"agent-a-{LIFECYCLE_RUN_ID}"
        agent_b_name = f"agent-b-{LIFECYCLE_RUN_ID}"

        # -- Step 1: Create two agents in the DB --
        resp_a = client.post(
            f"/api/v1/agents/create?name={agent_a_name}",
            headers=headers,
        )
        assert resp_a.status_code == 200, resp_a.text
        agent_a = resp_a.json()
        agent_a_key = agent_a["agentApiKey"]

        resp_b = client.post(
            f"/api/v1/agents/create?name={agent_b_name}",
            headers=headers,
        )
        assert resp_b.status_code == 200, resp_b.text
        agent_b = resp_b.json()
        agent_b_key = agent_b["agentApiKey"]

        # -- Step 2: Verify both appear in GET /list (DB agents) --
        list_resp = client.get("/api/v1/agents/list", headers=headers)
        assert list_resp.status_code == 200
        db_agents = list_resp.json()
        db_agent_ids = {a["agentId"] for a in db_agents}
        assert agent_a["agentId"] in db_agent_ids
        assert agent_b["agentId"] in db_agent_ids

        # -- Step 3: Verify GET /{agent_id} returns each agent --
        detail_a = client.get(f"/api/v1/agents/{agent_a['agentId']}", headers=headers)
        assert detail_a.status_code == 200
        assert detail_a.json()["agentId"] == agent_a["agentId"]

        detail_b = client.get(f"/api/v1/agents/{agent_b['agentId']}", headers=headers)
        assert detail_b.status_code == 200
        assert detail_b.json()["agentId"] == agent_b["agentId"]

        # -- Step 4: Register both agents (each authenticates with its API key) --
        reg_a = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name, "type": "api", "memory_mode": "cognee"},
        )
        assert reg_a.status_code == 201, reg_a.text
        connection_a_id = reg_a.json()["id"]
        assert agent_mode._active_count == 1

        reg_b = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_b_key},
            json={"agent_session_name": agent_b_name, "type": "sdk", "memory_mode": "hybrid"},
        )
        assert reg_b.status_code == 201, reg_b.text
        connection_b_id = reg_b.json()["id"]
        assert agent_mode._active_count == 2

        # -- Step 5: Re-registering same agent doesn't increment counter --
        reg_a_again = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name, "type": "api", "memory_mode": "cognee"},
        )
        assert reg_a_again.status_code == 201
        assert agent_mode._active_count == 2, "re-register should not increment"

        # -- Step 6: Verify GET /connections shows both --
        conn_resp = client.get("/api/v1/agents/connections", headers=headers)
        assert conn_resp.status_code == 200
        conn_ids = {a["id"] for a in conn_resp.json()["agents"]}
        assert connection_a_id in conn_ids
        assert connection_b_id in conn_ids

        # -- Step 7: Verify GET /connections/{agent_id} returns detail --
        detail_conn = client.get(
            f"/api/v1/agents/connections/{agent_a['agentId']}?agent_session_name={agent_a_name}",
            headers=headers,
        )
        assert detail_conn.status_code == 200
        assert detail_conn.json()["agent"]["agent_session_name"] == agent_a_name

        # -- Step 8: Unregister agent A's connection --
        unreg_a = client.post(
            "/api/v1/agents/unregister",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name},
        )
        assert unreg_a.status_code == 200
        assert unreg_a.json()["activeAgents"] == 1

        # -- Step 9: Active connections should show B but not A --
        conn_after = client.get("/api/v1/agents/connections", headers=headers)
        assert conn_after.status_code == 200
        conn_ids_after = {a["id"] for a in conn_after.json()["agents"]}
        assert connection_a_id not in conn_ids_after, "unregistered agent should not appear"
        assert connection_b_id in conn_ids_after, "still-registered agent should appear"

        # -- Step 9b: active_only=false should still show A as inactive --
        conn_all = client.get("/api/v1/agents/connections?active_only=false", headers=headers)
        assert conn_all.status_code == 200
        all_agents = conn_all.json()["agents"]
        all_ids = {a["id"] for a in all_agents}
        assert connection_a_id in all_ids, "inactive agent should appear with active_only=false"
        agent_a_conn = next(a for a in all_agents if a["id"] == connection_a_id)
        assert agent_a_conn["status"] == "inactive"

        # -- Step 10: DB list still shows both (unregister doesn't delete the user) --
        list_after = client.get("/api/v1/agents/list", headers=headers)
        db_ids_after = {a["agentId"] for a in list_after.json()}
        assert agent_a["agentId"] in db_ids_after
        assert agent_b["agentId"] in db_ids_after

        # -- Step 11: Unregister agent B's connection --
        unreg_b = client.post(
            "/api/v1/agents/unregister",
            headers={"X-Api-Key": agent_b_key},
            json={"agent_session_name": agent_b_name},
        )
        assert unreg_b.status_code == 200
        assert unreg_b.json()["activeAgents"] == 0

        # -- Step 12: Active connections should be empty --
        conn_empty = client.get("/api/v1/agents/connections", headers=headers)
        assert conn_empty.json()["agents"] == []

        # -- Step 12b: But all connections still exist as inactive in DB --
        conn_all_final = client.get("/api/v1/agents/connections?active_only=false", headers=headers)
        inactive_statuses = {a["status"] for a in conn_all_final.json()["agents"]}
        assert inactive_statuses == {"inactive"}

        # -- Step 13: Watchdog should trigger shutdown with 0 agents --
        with patch.object(agent_mode, "_shutdown_server") as mock_shutdown:
            agent_mode._watchdog()
            mock_shutdown.assert_called_once()

        # -- Step 14: Agent A can re-register after unregistering --
        re_reg = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name, "type": "api"},
        )
        assert re_reg.status_code == 201
        assert agent_mode._active_count == 1, "re-register after unregister should increment"

        # -- Step 15: Watchdog should NOT shutdown with active agent --
        with patch.object(agent_mode, "_shutdown_server") as mock_shutdown:
            agent_mode._watchdog()
            mock_shutdown.assert_not_called()

        # -- Step 16: Delete agent B from DB --
        del_resp = client.delete(
            f"/api/v1/agents/{agent_b['agentId']}",
            headers=headers,
        )
        assert del_resp.status_code == 200

        # -- Step 17: DB list no longer shows B --
        list_final = client.get("/api/v1/agents/list", headers=headers)
        final_ids = {a["agentId"] for a in list_final.json()}
        assert agent_b["agentId"] not in final_ids
        assert agent_a["agentId"] in final_ids

        # -- Step 18: Re-register agent A with different values (same name/type) --
        re_reg_updated = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={
                "agent_session_name": agent_a_name,
                "type": "api",
                "memory_mode": "session",
                "metadata": {"version": "2"},
            },
        )
        assert re_reg_updated.status_code == 201
        updated_conn = re_reg_updated.json()
        assert updated_conn["memory_mode"] == "session", "new values should replace old"
        assert updated_conn["metadata"]["version"] == "2"
        assert agent_mode._active_count == 1, "same user should not increment again"

        # -- Step 19: Only one connection for agent A in the registry --
        conn_after_update = client.get("/api/v1/agents/connections", headers=headers)
        agent_a_conns = [
            a for a in conn_after_update.json()["agents"] if a["id"] == updated_conn["id"]
        ]
        assert len(agent_a_conns) == 1
        assert agent_a_conns[0]["memory_mode"] == "session"

        # -- Step 20: Re-register with different name creates a second connection --
        re_reg_new_name = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={
                "agent_session_name": f"{agent_a_name}-v2",
                "type": "api",
                "memory_mode": "hybrid",
            },
        )
        assert re_reg_new_name.status_code == 201
        new_conn_id = re_reg_new_name.json()["id"]
        assert new_conn_id != updated_conn["id"], "different name should produce different ID"

        conn_both = client.get("/api/v1/agents/connections", headers=headers)
        both_ids = {a["id"] for a in conn_both.json()["agents"]}
        assert updated_conn["id"] in both_ids, "old connection should still exist"
        assert new_conn_id in both_ids, "new connection should also exist"
        assert agent_mode._active_count == 2, "two connections from same user count separately"

        # -- Step 21: Unregister one of agent A's connections, other stays active --
        client.post(
            "/api/v1/agents/unregister",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name},
        )

        conn_after_partial = client.get(
            "/api/v1/agents/connections?active_only=false", headers=headers
        )
        a_conn = next(
            (a for a in conn_after_partial.json()["agents"] if a["id"] == updated_conn["id"]),
            None,
        )
        assert a_conn is not None and a_conn["status"] == "inactive"
        v2_conn = next(
            (a for a in conn_after_partial.json()["agents"] if a["id"] == new_conn_id),
            None,
        )
        assert v2_conn is not None and v2_conn["status"] == "active", (
            "other connection should still be active"
        )

        # -- Step 22: Re-register agent A, verify DB status flips back to active --
        re_reg_after_deactivate = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_a_key},
            json={"agent_session_name": agent_a_name, "type": "api", "memory_mode": "cognee"},
        )
        assert re_reg_after_deactivate.status_code == 201
        reactivated_id = re_reg_after_deactivate.json()["id"]

        conn_reactivated = client.get(
            "/api/v1/agents/connections?active_only=false", headers=headers
        )
        reactivated = next(
            a for a in conn_reactivated.json()["agents"] if a["id"] == reactivated_id
        )
        assert reactivated["status"] == "active", "re-registered connection should be active in DB"

        # -- Step 23: Create agent C, register it, then delete it from DB --
        agent_c_name = f"agent-c-{LIFECYCLE_RUN_ID}"
        resp_c = client.post(
            f"/api/v1/agents/create?name={agent_c_name}",
            headers=headers,
        )
        assert resp_c.status_code == 200
        agent_c = resp_c.json()
        agent_c_key = agent_c["agentApiKey"]

        reg_c = client.post(
            "/api/v1/agents/register",
            headers={"X-Api-Key": agent_c_key},
            json={"agent_session_name": agent_c_name, "type": "api"},
        )
        assert reg_c.status_code == 201
        connection_c_id = reg_c.json()["id"]

        # Verify agent C shows in connections before delete
        conn_before_delete = client.get("/api/v1/agents/connections", headers=headers)
        assert connection_c_id in {a["id"] for a in conn_before_delete.json()["agents"]}

        # Delete agent C from DB
        del_c = client.delete(
            f"/api/v1/agents/{agent_c['agentId']}",
            headers=headers,
        )
        assert del_c.status_code == 200

        # Verify agent C is gone from DB list
        list_after_del = client.get("/api/v1/agents/list", headers=headers)
        assert agent_c["agentId"] not in {a["agentId"] for a in list_after_del.json()}

        # Verify agent C connections are cleaned up (not in active or inactive)
        conn_after_delete = client.get(
            "/api/v1/agents/connections?active_only=false", headers=headers
        )
        assert connection_c_id not in {a["id"] for a in conn_after_delete.json()["agents"]}, (
            "deleted agent's connections should be cleaned up"
        )


ISOLATION_RUN_ID = uuid.uuid4().hex[:8]
ISOLATION_USER_A_EMAIL = f"isolation-a-{ISOLATION_RUN_ID}@example.com"
ISOLATION_USER_B_EMAIL = f"isolation-b-{ISOLATION_RUN_ID}@example.com"
ISOLATION_PASSWORD = "isolation123!"


class TestMultiTenantIsolation:
    """Verify that users cannot see each other's agents or connections."""

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as c:
            yield c

    @pytest.fixture(scope="class")
    def user_a(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": ISOLATION_USER_A_EMAIL, "password": ISOLATION_PASSWORD},
        )
        assert reg.status_code in (200, 201), reg.text
        login = client.post(
            "/api/v1/auth/login",
            data={"username": ISOLATION_USER_A_EMAIL, "password": ISOLATION_PASSWORD},
        )
        assert login.status_code == 200
        return {"id": reg.json()["id"], "token": login.json()["access_token"]}

    @pytest.fixture(scope="class")
    def user_b(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": ISOLATION_USER_B_EMAIL, "password": ISOLATION_PASSWORD},
        )
        assert reg.status_code in (200, 201), reg.text
        login = client.post(
            "/api/v1/auth/login",
            data={"username": ISOLATION_USER_B_EMAIL, "password": ISOLATION_PASSWORD},
        )
        assert login.status_code == 200
        return {"id": reg.json()["id"], "token": login.json()["access_token"]}

    @pytest.fixture(autouse=True)
    def _reset(self):
        agent_mode._active_count = 0
        agent_mode._active_connection_ids.clear()
        agent_mode._watchdog_started = False
        clear_registered_agent_connections()
        yield

    def test_users_cannot_see_each_others_agents(self, client, user_a, user_b):
        headers_a = {"Authorization": f"Bearer {user_a['token']}"}
        headers_b = {"Authorization": f"Bearer {user_b['token']}"}

        # User A creates an agent
        resp_a = client.post(
            f"/api/v1/agents/create?name=agent-of-a-{ISOLATION_RUN_ID}",
            headers=headers_a,
        )
        assert resp_a.status_code == 200
        agent_a_id = resp_a.json()["agentId"]

        # User B creates an agent
        resp_b = client.post(
            f"/api/v1/agents/create?name=agent-of-b-{ISOLATION_RUN_ID}",
            headers=headers_b,
        )
        assert resp_b.status_code == 200
        agent_b_id = resp_b.json()["agentId"]

        # User A lists agents — should see only their own
        list_a = client.get("/api/v1/agents/list", headers=headers_a)
        a_agent_ids = {a["agentId"] for a in list_a.json()}
        assert agent_a_id in a_agent_ids
        assert agent_b_id not in a_agent_ids, "user A should not see user B's agent"

        # User B lists agents — should see only their own
        list_b = client.get("/api/v1/agents/list", headers=headers_b)
        b_agent_ids = {a["agentId"] for a in list_b.json()}
        assert agent_b_id in b_agent_ids
        assert agent_a_id not in b_agent_ids, "user B should not see user A's agent"

    def test_users_cannot_see_each_others_connections(self, client, user_a, user_b):
        headers_a = {"Authorization": f"Bearer {user_a['token']}"}
        headers_b = {"Authorization": f"Bearer {user_b['token']}"}

        # User A registers a connection
        reg_a = client.post(
            "/api/v1/agents/register",
            headers=headers_a,
            json={"agent_session_name": f"conn-a-{ISOLATION_RUN_ID}", "type": "api"},
        )
        assert reg_a.status_code == 201
        conn_a_id = reg_a.json()["id"]

        # User B registers a connection
        reg_b = client.post(
            "/api/v1/agents/register",
            headers=headers_b,
            json={"agent_session_name": f"conn-b-{ISOLATION_RUN_ID}", "type": "sdk"},
        )
        assert reg_b.status_code == 201
        conn_b_id = reg_b.json()["id"]

        # User A lists connections — should see only their own
        conns_a = client.get("/api/v1/agents/connections", headers=headers_a)
        a_conn_ids = {a["id"] for a in conns_a.json()["agents"]}
        assert conn_a_id in a_conn_ids
        assert conn_b_id not in a_conn_ids, "user A should not see user B's connection"

        # User B lists connections — should see only their own
        conns_b = client.get("/api/v1/agents/connections", headers=headers_b)
        b_conn_ids = {a["id"] for a in conns_b.json()["agents"]}
        assert conn_b_id in b_conn_ids
        assert conn_a_id not in b_conn_ids, "user B should not see user A's connection"

    def test_user_cannot_delete_others_agent(self, client, user_a, user_b):
        headers_a = {"Authorization": f"Bearer {user_a['token']}"}
        headers_b = {"Authorization": f"Bearer {user_b['token']}"}

        resp = client.post(
            f"/api/v1/agents/create?name=protected-{ISOLATION_RUN_ID}",
            headers=headers_a,
        )
        assert resp.status_code == 200
        agent_id = resp.json()["agentId"]

        del_resp = client.delete(
            f"/api/v1/agents/{agent_id}",
            headers=headers_b,
        )
        assert del_resp.status_code == 403, "user B should not be able to delete user A's agent"

    def test_user_cannot_get_others_agent(self, client, user_a, user_b):
        headers_a = {"Authorization": f"Bearer {user_a['token']}"}
        headers_b = {"Authorization": f"Bearer {user_b['token']}"}

        resp = client.post(
            f"/api/v1/agents/create?name=private-{ISOLATION_RUN_ID}",
            headers=headers_a,
        )
        assert resp.status_code == 200
        agent_id = resp.json()["agentId"]

        get_resp = client.get(
            f"/api/v1/agents/{agent_id}",
            headers=headers_b,
        )
        assert get_resp.status_code == 403, "user B should not be able to get user A's agent"
