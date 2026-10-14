"""AUTH_TOKEN_COOKIE_SECURE puts the Secure attribute on the auth cookie.

The cookie is the session; Secure keeps a browser from sending it over plain
http, where the token is readable on the path. It was hardcoded off, which
behind a TLS-terminating proxy could not be changed without editing code.
The transport reads the variable at import, so each case reloads it.
"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

TRANSPORT = "cognee.modules.users.authentication.default.default_transport"
ROUTER = "cognee.api.v1.users.routers.get_auth_router"


@pytest.fixture
def auth_app(monkeypatch, request):
    """An app with the auth router built under AUTH_TOKEN_COOKIE_SECURE=<param>."""
    monkeypatch.setenv("AUTH_TOKEN_COOKIE_SECURE", request.param)
    importlib.reload(importlib.import_module(TRANSPORT))
    router_module = importlib.reload(importlib.import_module(ROUTER))
    yield router_module
    monkeypatch.delenv("AUTH_TOKEN_COOKIE_SECURE")
    importlib.reload(importlib.import_module(TRANSPORT))
    importlib.reload(importlib.import_module(ROUTER))


def _attributes(set_cookie_header: str) -> set[str]:
    return {part.strip().split("=")[0].lower() for part in set_cookie_header.split(";")[1:]}


def _client(router_module, user) -> TestClient:
    app = FastAPI()
    app.include_router(router_module.get_auth_router(), prefix="/api/v1/auth")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return TestClient(app)


@pytest.mark.parametrize(
    "auth_app, secure", [("true", True), ("false", False)], indirect=["auth_app"]
)
def test_login_sets_secure_from_the_setting(auth_app, secure):
    user = MagicMock(email="a@b.test")
    strategy = MagicMock(write_token=AsyncMock(return_value="token-123"))
    backend = MagicMock(get_strategy=MagicMock(return_value=strategy))

    with (
        patch(f"{ROUTER}.authenticate_user", AsyncMock(return_value=user)),
        patch(f"{ROUTER}.get_client_auth_backend", return_value=backend),
    ):
        response = _client(auth_app, user).post(
            "/api/v1/auth/login", data={"username": "a@b.test", "password": "pw"}
        )

    assert response.status_code == 200, response.text
    header = response.headers["set-cookie"]
    assert header.startswith("auth_token=token-123;")
    assert ("secure" in _attributes(header)) is secure
    assert {"httponly", "samesite", "path"} <= _attributes(header)


@pytest.mark.parametrize("auth_app", ["true"], indirect=True)
def test_logout_deletes_with_the_attributes_login_set(auth_app):
    user = MagicMock(email="a@b.test")

    response = _client(auth_app, user).post("/api/v1/auth/logout")

    assert response.status_code == 200
    header = response.headers["set-cookie"]
    assert header.startswith('auth_token=""') and "Max-Age=0" in header
    assert {"secure", "httponly", "samesite", "path"} <= _attributes(header)
