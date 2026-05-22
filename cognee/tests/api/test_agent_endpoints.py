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
AGENT_WITH_PASSWORD = f"agent-pw-{RUN_ID}"
AGENT_WITHOUT_PASSWORD = f"agent-nopw-{RUN_ID}"
AGENT_PASSWORD = "agentpass456!"


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
    def agent_with_pw(self, client, owner_token):
        resp = client.post(
            f"/api/v1/agents/?name={AGENT_WITH_PASSWORD}&password={AGENT_PASSWORD}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 200
        return resp.json()

    @pytest.fixture(scope="class")
    def agent_without_pw(self, client, owner_token):
        resp = client.post(
            f"/api/v1/agents/?name={AGENT_WITHOUT_PASSWORD}",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 200
        return resp.json()

    def test_owner_login(self, client, owner):
        # Log out to clear cookies from the owner fixture
        client.post("/api/v1/auth/logout")

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": OWNER_EMAIL, "password": OWNER_PASSWORD},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_agent_with_password_can_login(self, client, owner, agent_with_pw):
        internal_email = f"{AGENT_WITH_PASSWORD}+{owner['id']}@cognee.agent"
        resp = client.post(
            "/api/v1/auth/login",
            data={"username": internal_email, "password": AGENT_PASSWORD},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_agent_without_password_cannot_login(self, client, owner, agent_without_pw):
        internal_email = f"{AGENT_WITHOUT_PASSWORD}+{owner['id']}@cognee.agent"

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": internal_email, "password": "anyguess"},
        )
        assert resp.status_code == 400
        assert "API key" in resp.json()["detail"]

    def test_agent_without_password_cannot_login_empty(self, client, owner, agent_without_pw):
        internal_email = f"{AGENT_WITHOUT_PASSWORD}+{owner['id']}@cognee.agent"

        resp = client.post(
            "/api/v1/auth/login",
            data={"username": internal_email, "password": ""},
        )
        assert resp.status_code == 400

    def test_agent_with_password_api_key_works(self, client, agent_with_pw):
        # Log out first to clear cookies
        client.post("/api/v1/auth/logout")

        api_key = agent_with_pw["agentApiKey"]
        resp = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": api_key},
        )
        assert resp.status_code == 200

    def test_agent_without_password_api_key_works(self, client, agent_without_pw):
        # Log out first to clear cookies
        client.post("/api/v1/auth/logout")

        api_key = agent_without_pw["agentApiKey"]
        resp = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": api_key},
        )
        assert resp.status_code == 200

    def test_list_agents_returns_both(self, client, owner_token, agent_with_pw, agent_without_pw):
        resp = client.get(
            "/api/v1/agents/",
            headers={"Authorization": f"Bearer {owner_token}"},
        )
        assert resp.status_code == 200
        agent_ids = {a["agentId"] for a in resp.json()}
        assert agent_with_pw["agentId"] in agent_ids
        assert agent_without_pw["agentId"] in agent_ids
