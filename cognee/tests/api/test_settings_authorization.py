import os
import pytest
from unittest.mock import patch

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["HASH_API_KEY"] = "false"

    from fastapi.testclient import TestClient


TEST_USER_EMAIL = "settings_test_user@example.com"
TEST_USER_PASSWORD = "securepassword123!"

DEFAULT_SUPERUSER_EMAIL = "default_user@example.com"
DEFAULT_SUPERUSER_PASSWORD = "default_password"


class TestSettingsAuthorization:
    """
    Regression test for CVE-2026-58473 / GHSA-49f7-whx5-4256.

    POST /api/v1/settings must require superuser privileges. A previous
    fix (#3115) was accidentally reverted; this test guards against that
    happening silently again.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as client:
            yield client

    def _get_auth_header(self, client, email, password):
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": email, "password": password},
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    def test_non_superuser_cannot_modify_settings(self, client):
        # Register a normal (non-superuser) user
        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
        )
        assert register_response.status_code in (201, 400)

        headers = self._get_auth_header(client, TEST_USER_EMAIL, TEST_USER_PASSWORD)

        response = client.post(
            "/api/v1/settings",
            json={"llm": {"provider": "openai", "model": "gpt-4o-mini", "api_key": "attacker-key"}},
            headers=headers,
        )

        assert response.status_code == 403, (
            "Non-superuser was able to modify global settings — CVE-2026-58473 regression!"
        )

    def test_superuser_can_modify_settings(self, client):
        headers = self._get_auth_header(
            client, DEFAULT_SUPERUSER_EMAIL, DEFAULT_SUPERUSER_PASSWORD
        )

        response = client.get("/api/v1/settings", headers=headers)
        assert response.status_code == 200

        current = response.json()
        response = client.post(
            "/api/v1/settings",
            json={
                "llm": {
                    "provider": current["llm"]["provider"],
                    "model": current["llm"]["model"],
                    "api_key": "test-key-not-attacker-controlled",
                }
            },
            headers=headers,
        )
        assert response.status_code == 200, (
            f"Superuser should still be able to modify settings. Got: {response.text}"
        )