"""AUTH_TOKEN_COOKIE_SECURE puts the Secure attribute on the auth cookie.

The cookie is the session; Secure keeps a browser from sending it over plain
http, where the token is readable on the path. It was hardcoded off, which
behind a TLS-terminating proxy could not be changed without editing code.
The contract: the setting decides Secure on login, a value other than true or
false raises, and logout deletes the cookie with the attributes login set.
The transport reads the variable at import, so the cases load a private copy of
the module instead of reloading the one the app's auth backend holds.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.users.routers.get_auth_router import get_auth_router
from cognee.modules.users.methods import get_authenticated_user

# The routers package re-exports get_auth_router under the module's name, so a dotted
# patch target resolves to the function on Python 3.10; patch the module object instead.
ROUTER = importlib.import_module("cognee.api.v1.users.routers.get_auth_router")
# The package re-exports the object under the module's name, so look the module up by name.
TRANSPORT_FILE = importlib.import_module(
    "cognee.modules.users.authentication.default.default_transport"
).__file__


def _transport_with_env(monkeypatch, value, domain=None):
    """A fresh CookieTransport built the way the module builds it, under the given env."""
    for name, setting in (
        ("AUTH_TOKEN_COOKIE_SECURE", value),
        ("AUTH_TOKEN_COOKIE_DOMAIN", domain),
    ):
        if setting is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, setting)
    spec = importlib.util.spec_from_file_location("_transport_copy", TRANSPORT_FILE)
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    return copy.default_transport


def _attributes(set_cookie_header: str) -> set[str]:
    return {part.strip().split("=")[0].lower() for part in set_cookie_header.split(";")[1:]}


def _client(user) -> TestClient:
    app = FastAPI()
    app.include_router(get_auth_router(), prefix="/api/v1/auth")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return TestClient(app)


@pytest.mark.parametrize(
    "value, expected", [(None, False), ("false", False), ("true", True), (" TRUE ", True)]
)
def test_the_transport_reads_secure_from_the_environment(monkeypatch, value, expected):
    """The default is off, and true or false is accepted in any case, padded or not."""
    assert _transport_with_env(monkeypatch, value).cookie_secure is expected


@pytest.mark.parametrize("value", ["1", "yes", "on", "enabled", ""])
def test_the_transport_refuses_any_other_value(monkeypatch, value):
    """A typo must not leave the session cookie without Secure and say nothing."""
    with pytest.raises(ValueError, match="AUTH_TOKEN_COOKIE_SECURE"):
        _transport_with_env(monkeypatch, value)


@pytest.mark.parametrize("value, secure", [("true", True), ("false", False)])
def test_login_sets_secure_from_the_environment(monkeypatch, value, secure):
    """The login route copies cookie_secure, read from the environment, onto the cookie."""
    transport = _transport_with_env(monkeypatch, value, domain="x.test")
    transport.cookie_path = "/api"
    user = MagicMock(email="a@b.test")
    strategy = MagicMock(write_token=AsyncMock(return_value="token-123"))
    backend = MagicMock(get_strategy=MagicMock(return_value=strategy))

    with (
        patch.object(ROUTER, "default_transport", transport),
        patch.object(ROUTER, "authenticate_user", AsyncMock(return_value=user)),
        patch.object(ROUTER, "get_client_auth_backend", return_value=backend),
    ):
        response = _client(user).post(
            "/api/v1/auth/login", data={"username": "a@b.test", "password": "pw"}
        )

    assert response.status_code == 200, response.text
    header = response.headers["set-cookie"]
    assert header.startswith("auth_token=token-123;")
    assert ("secure" in _attributes(header)) is secure
    assert "httponly" in _attributes(header), header
    assert "SameSite=Lax" in header, header
    assert "Path=/api" in header and "Domain=x.test" in header, header
    assert "max-age" not in _attributes(header), "login writes a session cookie"


def test_logout_deletes_with_the_attributes_login_set(monkeypatch):
    """A browser keeps a cookie that a deletion with another path or domain does not match.

    Path and domain are set to non-default values, since the defaults are what
    delete_cookie emits anyway.
    """
    transport = _transport_with_env(monkeypatch, "true")
    transport.cookie_path = "/api"
    transport.cookie_domain = "x.test"
    user = MagicMock(email="a@b.test")

    with patch.object(ROUTER, "default_transport", transport):
        response = _client(user).post("/api/v1/auth/logout")

    assert response.status_code == 200
    header = response.headers["set-cookie"]
    assert header.startswith('auth_token=""') and "Max-Age=0" in header
    assert {"secure", "httponly"} <= _attributes(header), header
    assert "SameSite=Lax" in header, header
    assert "Path=/api" in header and "Domain=x.test" in header, header


@pytest.mark.parametrize("value, expected", [("x.test", "x.test"), ("", None)])
def test_the_transport_reads_the_cookie_domain_from_the_environment(monkeypatch, value, expected):
    """An empty AUTH_TOKEN_COOKIE_DOMAIN means no Domain attribute, not an empty one."""
    assert _transport_with_env(monkeypatch, None, domain=value).cookie_domain == expected


@pytest.mark.parametrize("value", ["true", "false"])
def test_the_app_uses_one_transport_that_follows_the_environment(value):
    """The router and the auth backend share the real transport, built from the environment.

    Runs in a subprocess, since the real module reads the variable once at import.
    """
    code = (
        "import importlib;"
        "from cognee.modules.users.authentication.get_client_auth_backend import "
        "get_client_auth_backend;"
        "router = importlib.import_module('cognee.api.v1.users.routers.get_auth_router');"
        "transport = get_client_auth_backend().transport;"
        "assert transport is router.default_transport;"
        f"assert transport.cookie_secure is {value.capitalize()}"
    )
    env = {**os.environ, "AUTH_TOKEN_COOKIE_SECURE": value}

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
