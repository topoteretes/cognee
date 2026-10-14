"""The auth cookie's ``Secure`` attribute follows AUTH_TOKEN_COOKIE_SECURE.

The cookie is the session. ``Secure`` keeps a browser from sending it over
plain http, where the token is readable on the path. It was hardcoded off,
which behind a TLS-terminating proxy (cognee sees only http) could not be
changed without editing code. The setting is a tri-state: true, false, or
unset for "https requests only", as cognee sees the scheme.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.users.routers.get_auth_router import get_auth_router
from cognee.modules.users.authentication.default.default_transport import cookie_secure_for
from cognee.modules.users.methods import get_authenticated_user

ROUTER = "cognee.api.v1.users.routers.get_auth_router"


def _setting(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("AUTH_TOKEN_COOKIE_SECURE", raising=False)
    else:
        monkeypatch.setenv("AUTH_TOKEN_COOKIE_SECURE", value)


def _request(scheme: str):
    request = MagicMock()
    request.url.scheme = scheme
    return request


@pytest.mark.parametrize(
    "value, http, https",
    [
        (None, False, True),
        ("", False, True),
        ("auto", False, True),
        ("true", True, True),
        ("1", True, True),
        ("false", False, False),
        ("0", False, False),
    ],
)
def test_secure_follows_the_setting_then_the_request_scheme(monkeypatch, value, http, https):
    _setting(monkeypatch, value)

    assert cookie_secure_for(_request("http")) is http
    assert cookie_secure_for(_request("https")) is https


def _client(user, base_url: str) -> TestClient:
    app = FastAPI()
    app.include_router(get_auth_router(), prefix="/api/v1/auth")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return TestClient(app, base_url=base_url)


def _attributes(set_cookie_header: str) -> set[str]:
    """The attribute names of a Set-Cookie header, lower-cased."""
    return {part.strip().split("=")[0].lower() for part in set_cookie_header.split(";")[1:]}


@pytest.mark.parametrize(
    "value, base_url, secure",
    [
        (None, "http://testserver", False),
        (None, "https://testserver", True),
        ("true", "http://testserver", True),
        ("false", "https://testserver", False),
    ],
)
def test_login_writes_the_cookie_with_the_resolved_secure_flag(
    monkeypatch, value, base_url, secure
):
    _setting(monkeypatch, value)
    user = MagicMock(email="a@b.test")
    strategy = MagicMock(write_token=AsyncMock(return_value="token-123"))
    backend = MagicMock(get_strategy=MagicMock(return_value=strategy))

    with (
        patch(f"{ROUTER}.authenticate_user", AsyncMock(return_value=user)),
        patch(f"{ROUTER}.get_client_auth_backend", return_value=backend),
    ):
        response = _client(user, base_url).post(
            "/api/v1/auth/login", data={"username": "a@b.test", "password": "pw"}
        )

    assert response.status_code == 200, response.text
    header = response.headers["set-cookie"]
    assert header.startswith("auth_token=token-123;")
    attributes = _attributes(header)
    assert ("secure" in attributes) is secure
    assert {"httponly", "samesite", "path"} <= attributes


def test_logout_deletes_with_the_same_attributes_it_was_set_with(monkeypatch):
    _setting(monkeypatch, "true")
    user = MagicMock(email="a@b.test")

    response = _client(user, "http://testserver").post("/api/v1/auth/logout")

    assert response.status_code == 200
    header = response.headers["set-cookie"]
    assert header.startswith('auth_token=""') and "Max-Age=0" in header
    assert {"secure", "httponly", "samesite", "path"} <= _attributes(header)
