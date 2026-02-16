"""Tests for token refresh and revoke endpoints (POST /auth/refresh, POST /auth/revoke)."""

import asyncio
import os
import pathlib

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import cognee
from cognee.api.client import app
from cognee.modules.users.methods import create_user, get_default_user

pytestmark = pytest.mark.asyncio


@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for our test module."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def client():
    """Create an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture(scope="module")
async def token_refresh_env():
    """Set up environment and ensure default user exists for auth tests."""
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )

    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "True")
    base_dir = pathlib.Path(__file__).parent
    data_dir = base_dir / ".data_storage/test_token_refresh"
    system_dir = base_dir / ".cognee_system/test_token_refresh"
    (system_dir / "databases").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cognee.config.data_root_directory(str(data_dir))
    cognee.config.system_root_directory(str(system_dir))
    create_relational_engine.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    from cognee.infrastructure.databases.relational import get_relational_engine

    db_engine = get_relational_engine()
    await db_engine.create_database()
    await get_default_user()
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _login(client: AsyncClient, email: str, password: str) -> dict:
    """Login and return JSON response (access_token, refresh_token, etc.)."""
    response = await client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
        timeout=15,
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_login_returns_refresh_token(client: AsyncClient, token_refresh_env: None):
    """Login response must include access_token and refresh_token."""
    data = await _login(client, "default_user@example.com", "default_password")
    assert "access_token" in data
    assert "refresh_token" in data
    assert data.get("token_type") == "bearer"
    assert "expires_in" in data


async def test_refresh_returns_new_tokens(client: AsyncClient, token_refresh_env: None):
    """POST /auth/refresh with valid refresh_token returns new access_token and refresh_token."""
    login_data = await _login(client, "default_user@example.com", "default_password")
    refresh_token = login_data["refresh_token"]
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=15,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["refresh_token"] != refresh_token  # rotation: new token
    assert data.get("token_type") == "bearer"


async def test_refresh_with_invalid_token_returns_401(client: AsyncClient, token_refresh_env: None):
    """POST /auth/refresh with invalid or expired refresh_token returns 401."""
    response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": "invalid-token"},
        timeout=15,
    )
    assert response.status_code == 401
    assert "Invalid or expired refresh token" in response.json().get("detail", "")


async def test_refresh_after_revoke_fails(client: AsyncClient, token_refresh_env: None):
    """After POST /auth/revoke, using the previous refresh_token returns 401."""
    login_data = await _login(client, "default_user@example.com", "default_password")
    access_token = login_data["access_token"]
    refresh_token = login_data["refresh_token"]

    # Revoke all refresh tokens for the user
    revoke_response = await client.post(
        "/api/v1/auth/revoke",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    assert revoke_response.status_code == 200
    assert revoke_response.json().get("revoked", 0) >= 1

    # Using the old refresh token should fail
    refresh_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=15,
    )
    assert refresh_response.status_code == 401


async def test_new_access_token_works_after_refresh(client: AsyncClient, token_refresh_env: None):
    """New access_token from /auth/refresh can be used to call protected endpoints."""
    login_data = await _login(client, "default_user@example.com", "default_password")
    refresh_token = login_data["refresh_token"]

    refresh_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
        timeout=15,
    )
    assert refresh_response.status_code == 200
    new_access_token = refresh_response.json()["access_token"]

    # Use new access token for /me
    me_response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {new_access_token}"},
        timeout=15,
    )
    assert me_response.status_code == 200
    assert me_response.json().get("email") == "default_user@example.com"
