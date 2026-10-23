"""AUTH_TOKEN_COOKIE_SECURE puts the Secure attribute on the auth cookie.

The cookie is the session; Secure keeps a browser from sending it over plain
http, where the token is readable on the path. It was hardcoded off, which
behind a TLS-terminating proxy could not be changed without editing code.
The transport reads the variable at import through secure_cookie_from_env, tested on its
own; the route cases set the attribute it feeds.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.users.routers.get_auth_router import get_auth_router
from cognee.modules.users.authentication.default.default_transport import (
    default_transport,
    secure_cookie_from_env,
)
from cognee.modules.users.methods import get_authenticated_user

ROUTER = "cognee.api.v1.users.routers.get_auth_router"


@pytest.fixture
def auth_app(monkeypatch, request):
    """Sets the shared transport's cookie_secure to the param."""
    monkeypatch.setattr(default_transport, "cookie_secure", request.param)


def _attributes(set_cookie_header: str) -> set[str]:
    return {part.strip().split("=")[0].lower() for part in set_cookie_header.split(";")[1:]}


def _client(user) -> TestClient:
    app = FastAPI()
    app.include_router(get_auth_router(), prefix="/api/v1/auth")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return TestClient(app)


@pytest.mark.parametrize("auth_app, secure", [(True, True), (False, False)], indirect=["auth_app"])
def test_login_sets_secure_from_the_setting(auth_app, secure):
    user = MagicMock(email="a@b.test")
    strategy = MagicMock(write_token=AsyncMock(return_value="token-123"))
    backend = MagicMock(get_strategy=MagicMock(return_value=strategy))

    with (
        patch(f"{ROUTER}.authenticate_user", AsyncMock(return_value=user)),
        patch(f"{ROUTER}.get_client_auth_backend", return_value=backend),
    ):
        response = _client(user).post(
            "/api/v1/auth/login", data={"username": "a@b.test", "password": "pw"}
        )

    assert response.status_code == 200, response.text
    header = response.headers["set-cookie"]
    assert header.startswith("auth_token=token-123;")
    assert ("secure" in _attributes(header)) is secure
    assert {"httponly", "samesite", "path"} <= _attributes(header)


@pytest.mark.parametrize("auth_app", [True], indirect=True)
def test_logout_deletes_with_the_attributes_login_set(auth_app, monkeypatch):
    monkeypatch.setattr(default_transport, "cookie_path", "/api")
    monkeypatch.setattr(default_transport, "cookie_domain", "x.test")
    user = MagicMock(email="a@b.test")

    response = _client(user).post("/api/v1/auth/logout")

    assert response.status_code == 200
    header = response.headers["set-cookie"]
    assert header.startswith('auth_token=""') and "Max-Age=0" in header
    assert {"secure", "httponly", "samesite"} <= _attributes(header)
    assert "Path=/api" in header and "Domain=x.test" in header


@pytest.mark.parametrize(
    "value, expected", [(None, False), ("false", False), ("true", True), (" TRUE ", True)]
)
def test_secure_cookie_from_env(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("AUTH_TOKEN_COOKIE_SECURE", raising=False)
    else:
        monkeypatch.setenv("AUTH_TOKEN_COOKIE_SECURE", value)

    assert secure_cookie_from_env() is expected


@pytest.mark.parametrize("value", ["1", "yes", "on", "enabled", ""])
def test_secure_cookie_from_env_rejects_other_values(monkeypatch, value):
    monkeypatch.setenv("AUTH_TOKEN_COOKIE_SECURE", value)

    with pytest.raises(ValueError, match="AUTH_TOKEN_COOKIE_SECURE"):
        secure_cookie_from_env()
