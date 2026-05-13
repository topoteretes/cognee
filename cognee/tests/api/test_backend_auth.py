import os
import uuid
import asyncio
import pytest
from unittest.mock import patch

with patch("dotenv.load_dotenv"):
    os.environ["REQUIRE_AUTHENTICATION"] = "true"
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    os.environ["HASH_API_KEY"] = "false"

    from fastapi.testclient import TestClient


TEST_USER_EMAIL = "apikey_test_user@example.com"
TEST_USER_PASSWORD = "securepassword123!"

HASH_TEST_USER_EMAIL = "hash_test_user@example.com"
HASH_TEST_USER_PASSWORD = "securepassword123!"


class TestAuthFlow:
    """
    End-to-end test: register account, login with JWT, create API key,
    and confirm a request authenticated with that API key succeeds.
    """

    @pytest.fixture(scope="class")
    def client(self):
        from cognee.api.client import app

        with TestClient(app) as client:
            yield client

    def test_register_login_create_api_key_and_authenticate(self, client):
        # Register a new user (ignore if already exists)
        register_response = client.post(
            "/api/v1/auth/register",
            json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
        )
        assert register_response.status_code in (201, 400)

        # Login and retrieve JWT bearer token
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD},
        )
        assert login_response.status_code == 200
        access_token = login_response.json()["access_token"]
        assert access_token

        # Create an API key using the bearer token
        # Note: There is a maximum number of API keys allowed per user (10)
        create_key_response = client.post(
            "/api/v1/auth/api-keys",
            json={"name": "integration test key"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert create_key_response.status_code == 200
        api_key_id = create_key_response.json()["id"]
        api_key = create_key_response.json()["key"]
        assert api_key

        # Test if Cookie authentication works on an authenticated endpoint
        me_response = client.get("/api/v1/auth/me")
        assert me_response.status_code == 200

        # Note: we have to log out so Cookies don't interfere with API key authentication in the next step
        client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {access_token}"})

        # Test if Cookie authentication doesn't work anymore after logging out
        me_response = client.get("/api/v1/auth/me")
        assert me_response.status_code == 401

        # Use only the API key on an authenticated endpoint and confirm it succeeds
        me_response = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": api_key},
        )
        assert me_response.status_code == 200
        assert me_response.json()["email"] == TEST_USER_EMAIL

        delete_response = client.delete(
            f"/api/v1/auth/api-keys/{api_key_id}", headers={"X-Api-Key": api_key}
        )

        assert delete_response.status_code == 200, "API key deletion should succeed"

        me_response = client.get(
            "/api/v1/auth/me",
            headers={"X-Api-Key": api_key},
        )
        assert me_response.status_code == 401, "Deleted API key should no longer authenticate"

        # Test if Bearer token works after logging out and deleting API key
        me_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert me_response.status_code == 200, (
            "Bearer token should still work after logging out and deleting API key"
        )


class TestHashApiKey:
    """
    Confirm that when HASH_API_KEY=true, the API key stored in the database
    is the SHA-256 hash of the raw key, not the raw key itself.
    """

    @pytest.fixture(scope="class")
    def client(self):
        with patch("cognee.modules.users.api_key.hash_api_key.HASH_API_KEY", True):
            from cognee.api.client import app

            with TestClient(app) as client:
                yield client

    def test_api_key_is_stored_as_hash(self, client):
        from sqlalchemy import select
        from cognee.modules.users.models.UserApiKey import UserApiKey
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.users.api_key.hash_api_key import hash_api_key as compute_hash

        # Register (ignore if already exists) and login
        client.post(
            "/api/v1/auth/register",
            json={"email": HASH_TEST_USER_EMAIL, "password": HASH_TEST_USER_PASSWORD},
        )
        login_response = client.post(
            "/api/v1/auth/login",
            data={"username": HASH_TEST_USER_EMAIL, "password": HASH_TEST_USER_PASSWORD},
        )
        assert login_response.status_code == 200
        access_token = login_response.json()["access_token"]

        # Create an API key with HASH_API_KEY=True active
        # Note: There is a maximum number of API keys allowed per user (10)
        with patch("cognee.modules.users.api_key.hash_api_key.HASH_API_KEY", True):
            create_key_response = client.post(
                "/api/v1/auth/api-keys",
                json={"name": "hash verification key"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
        assert create_key_response.status_code == 200

        raw_key = create_key_response.json()["key"]
        key_id = uuid.UUID(create_key_response.json()["id"])

        # Query the database directly to verify the stored value is hashed
        async def get_stored_api_key():
            engine = get_relational_engine()
            async with engine.get_async_session() as session:
                result = (await session.execute(select(UserApiKey).filter_by(id=key_id))).scalar()
                return result.api_key

        stored_key = asyncio.run(get_stored_api_key())

        assert stored_key != raw_key, "Raw key should not be stored in plaintext"
        assert stored_key == compute_hash(raw_key), (
            "Stored key should be SHA-256 hash of the raw key"
        )
