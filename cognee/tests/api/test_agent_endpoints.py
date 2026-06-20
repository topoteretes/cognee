import os
import uuid
import pytest
from unittest.mock import patch

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["HASH_API_KEY"] = "false"

    from fastapi.testclient import TestClient

RUN_ID = uuid.uuid4().hex[:8]
OWNER_EMAIL = f"agent-owner-{RUN_ID}@example.com"
OWNER_PASSWORD = "ownerpass123!"
AGENT_NAME = f"test-agent-{RUN_ID}"


class TestAgentEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as client:
            yield client

    @pytest.fixture(scope="class")
    def owner(self, client):
        reg = client.post(
            "/api/v1/auth/register",
            json={"email": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert reg.status_code in (200, 201)
        owner_id = reg.json()["id"]

        login = client.post(
            "/api/v1/auth/login",
            data={"username": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert login.status_code == 200
        return {"id": owner_id, "token": login.json()["access_token"]}

    @pytest.fixture(scope="class")
    def owner_token(self, owner):
        return owner["token"]

    @pytest.fixture(scope="class")
    def agent(self, client, owner_token):
        resp = client.post(
            f"/api/v1/agents/create?name={AGENT_NAME}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 200
        return resp.json()

    def test_owner_login(self, client, owner):
        client.post("/api/v1/auth/logout")

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_agent_cannot_login(self, client, owner, agent):
        internal_email = f"{AGENT_NAME}+{owner['id']}@cognee.agent"

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": internal_email, "password": "anyguess"},
        )
        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]

    def test_agent_cannot_login_empty_password(self, client, owner, agent):
        internal_email = f"{AGENT_NAME}+{owner['id']}@cognee.agent"

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": internal_email, "password": ""},
        )
        assert resp.status_code == 400

    def test_agent_api_key_works(self, client, agent):
        client.post("/api/v1/auth/logout")

        resp = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": agent["agentApiKey"]},
        )
        assert resp.status_code == 200

    def test_list_agents_returns_agent(self, client, owner_token, agent):
        resp = client.get(
            "/api/v1/agents/list",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 200
        agents = resp.json()
        assert isinstance(agents, list)
        agent_ids = {a["agentId"] for a in agents}
        assert agent["agentId"] in agent_ids

    def test_duplicate_agent_returns_409(self, client, owner_token, agent):
        resp = client.post(
            f"/api/v1/agents/create?name={AGENT_NAME}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 409
