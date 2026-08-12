"""API-key bootstrap: serve(url=...) must work against a fresh server.

A fresh local server has no API key to hand out-of-band, so serve()
resolves one the way the agent integrations do: reuse the key saved for
this URL, else log in as the (default) user and mint one over the JWT
session. Only when both fail may the connection proceed keyless (a
server running with authentication off).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.serve import local_auth
from cognee.api.v1.serve.exceptions import CogneeAuthError
from cognee.api.v1.serve.local_auth import login_and_mint_api_key, resolve_bootstrap_credentials


class FakeResponse:
    def __init__(self, status=200, json_body=None, text_body=""):
        self.status = status
        self._json = json_body if json_body is not None else {}
        self._text = text_body

    async def json(self):
        return self._json

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeAuthSession:
    """Routes (method, path-suffix) to canned responses and records calls."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def _dispatch(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        for (handler_method, suffix), response in self.handlers.items():
            if handler_method == method and url.endswith(suffix):
                return response
        raise AssertionError(f"unexpected request: {method} {url}")

    def post(self, url, **kwargs):
        return self._dispatch("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._dispatch("GET", url, **kwargs)


@pytest.fixture
def fake_auth_session(monkeypatch):
    """Patch aiohttp.ClientSession inside local_auth with a routable fake."""
    holder = {}

    def install(handlers):
        session = FakeAuthSession(handlers)
        monkeypatch.setattr(local_auth.aiohttp, "ClientSession", lambda timeout=None: session)
        holder["session"] = session
        return session

    return install


LOGIN_OK = FakeResponse(json_body={"access_token": "jwt-token"})


def test_mint_reuses_first_usable_existing_key(fake_auth_session):
    session = fake_auth_session(
        {
            ("POST", "/api/v1/auth/login"): LOGIN_OK,
            ("GET", "/api/v1/auth/api-keys"): FakeResponse(
                json_body=[{"key": "ck_existing", "name": "older"}]
            ),
        }
    )
    key = asyncio.run(login_and_mint_api_key("http://localhost:8011"))
    assert key == "ck_existing"
    login_call = session.calls[0]
    assert login_call["data"] == {
        "username": "default_user@example.com",
        "password": "default_password",
    }
    # The JWT authenticates the api-keys call via the auth_token cookie.
    assert session.calls[1]["cookies"] == {"auth_token": "jwt-token"}


def test_mint_skips_masked_keys_and_creates_new_one(fake_auth_session):
    session = fake_auth_session(
        {
            ("POST", "/api/v1/auth/login"): LOGIN_OK,
            ("GET", "/api/v1/auth/api-keys"): FakeResponse(
                json_body=[{"key": "************", "name": "hashed"}]
            ),
            ("POST", "/api/v1/auth/api-keys"): FakeResponse(json_body={"key": "ck_minted"}),
        }
    )
    key = asyncio.run(login_and_mint_api_key("http://localhost:8011"))
    assert key == "ck_minted"
    create_call = session.calls[-1]
    assert create_call["json"] == {"name": "cognee-serve-bootstrap"}


def test_login_rejection_raises_auth_error(fake_auth_session):
    fake_auth_session(
        {
            ("POST", "/api/v1/auth/login"): FakeResponse(
                status=400, text_body="LOGIN_BAD_CREDENTIALS"
            ),
        }
    )
    with pytest.raises(CogneeAuthError) as excinfo:
        asyncio.run(login_and_mint_api_key("http://localhost:8011"))
    assert excinfo.value.operation == "login"


def test_bootstrap_credentials_prefer_plugin_env_vars(monkeypatch):
    monkeypatch.setenv("COGNEE_USER_EMAIL", "owner@corp.dev")
    monkeypatch.setenv("COGNEE_USER_PASSWORD", "hunter2")
    monkeypatch.setenv("DEFAULT_USER_EMAIL", "server@corp.dev")
    assert resolve_bootstrap_credentials() == ("owner@corp.dev", "hunter2")


def test_bootstrap_credentials_fall_back_to_server_defaults(monkeypatch):
    monkeypatch.delenv("COGNEE_USER_EMAIL", raising=False)
    monkeypatch.delenv("COGNEE_USER_PASSWORD", raising=False)
    monkeypatch.setenv("DEFAULT_USER_EMAIL", "server@corp.dev")
    monkeypatch.delenv("DEFAULT_USER_PASSWORD", raising=False)
    assert resolve_bootstrap_credentials() == ("server@corp.dev", "default_password")


# ----- serve(url=...) integration -----


@pytest.fixture
def isolated_credentials(tmp_path, monkeypatch):
    from cognee.api.v1.serve import credentials

    path = tmp_path / "cloud_credentials.json"
    monkeypatch.setattr(credentials, "_CREDENTIALS_FILE", path)
    return path


@pytest.fixture
def quiet_client(monkeypatch):
    from cognee.api.v1.serve.cloud_client import CloudClient

    monkeypatch.setattr(CloudClient, "_health_check", AsyncMock(return_value=True))


@pytest.fixture(autouse=True)
def reset_remote_client():
    from cognee.api.v1.serve.state import set_remote_client

    yield
    set_remote_client(None)


def test_serve_direct_mints_key_when_none_given(isolated_credentials, quiet_client, monkeypatch):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    monkeypatch.setattr(
        local_auth_module, "login_and_mint_api_key", AsyncMock(return_value="ck_minted")
    )
    client = asyncio.run(_serve_direct("http://localhost:8011"))
    assert client.api_key == "ck_minted"
    # The minted key is persisted so the next serve() reconnects directly.
    assert '"api_key": "ck_minted"' in isolated_credentials.read_text()


def test_serve_direct_reuses_saved_key_for_same_url(
    isolated_credentials, quiet_client, monkeypatch
):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials
    from cognee.api.v1.serve.serve import _serve_direct

    save_credentials(
        CloudCredentials(access_token="", service_url="http://localhost:8011", api_key="ck_saved")
    )
    mint = AsyncMock(return_value="ck_should_not_be_used")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)

    client = asyncio.run(_serve_direct("http://localhost:8011"))
    assert client.api_key == "ck_saved"
    mint.assert_not_awaited()


def test_serve_direct_ignores_saved_key_for_other_url(
    isolated_credentials, quiet_client, monkeypatch
):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials
    from cognee.api.v1.serve.serve import _serve_direct

    save_credentials(
        CloudCredentials(access_token="", service_url="http://localhost:8000", api_key="ck_other")
    )
    monkeypatch.setattr(
        local_auth_module, "login_and_mint_api_key", AsyncMock(return_value="ck_minted")
    )
    client = asyncio.run(_serve_direct("http://localhost:8011"))
    assert client.api_key == "ck_minted"


def test_serve_direct_connects_keyless_when_bootstrap_fails(
    isolated_credentials, quiet_client, monkeypatch
):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    monkeypatch.setattr(
        local_auth_module,
        "login_and_mint_api_key",
        AsyncMock(side_effect=CogneeAuthError("denied", status=400, body="", operation="login")),
    )
    client = asyncio.run(_serve_direct("http://localhost:8011"))
    assert client.api_key == ""


def test_explicit_api_key_skips_bootstrap(isolated_credentials, quiet_client, monkeypatch):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    mint = AsyncMock(return_value="ck_should_not_be_used")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)
    client = asyncio.run(_serve_direct("http://localhost:8011", "ck_explicit"))
    assert client.api_key == "ck_explicit"
    mint.assert_not_awaited()


# ----- host gate: the login flow must not run against arbitrary hosts -----


def test_private_hosts_are_recognized():
    from cognee.api.v1.serve.local_auth import is_private_host

    for url in (
        "http://localhost:8011",
        "http://app.localhost:8011",
        "http://127.0.0.1:8000",
        "http://[::1]:8000",
        "http://10.0.0.7:8000",
        "http://172.16.4.2:8000",
        "http://192.168.1.5:8000",
    ):
        assert is_private_host(url), url

    for url in (
        "https://tenant-abc.cloud.cognee.ai",
        "https://api.cognee.ai",
        "http://8.8.8.8:8000",
        # A DNS name can resolve anywhere, so it never counts as private.
        "http://my-internal-server:8000",
    ):
        assert not is_private_host(url), url


def test_no_mint_against_remote_host_by_default(isolated_credentials, quiet_client, monkeypatch):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    monkeypatch.delenv("COGNEE_AUTH_BOOTSTRAP", raising=False)
    mint = AsyncMock(return_value="ck_should_not_be_used")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)

    client = asyncio.run(_serve_direct("https://tenant-abc.cloud.cognee.ai"))
    assert client.api_key == ""
    mint.assert_not_awaited()


def test_explicit_opt_in_allows_mint_against_remote_host(
    isolated_credentials, quiet_client, monkeypatch
):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    mint = AsyncMock(return_value="ck_minted")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)

    client = asyncio.run(_serve_direct("https://my.cognee.corp", bootstrap_auth=True))
    assert client.api_key == "ck_minted"


def test_env_flag_allows_mint_against_remote_host(isolated_credentials, quiet_client, monkeypatch):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    monkeypatch.setenv("COGNEE_AUTH_BOOTSTRAP", "true")
    mint = AsyncMock(return_value="ck_minted")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)

    client = asyncio.run(_serve_direct("https://my.cognee.corp"))
    assert client.api_key == "ck_minted"


def test_explicit_false_disables_mint_even_for_localhost(
    isolated_credentials, quiet_client, monkeypatch
):
    from cognee.api.v1.serve import local_auth as local_auth_module
    from cognee.api.v1.serve.serve import _serve_direct

    mint = AsyncMock(return_value="ck_should_not_be_used")
    monkeypatch.setattr(local_auth_module, "login_and_mint_api_key", mint)

    client = asyncio.run(_serve_direct("http://localhost:8011", bootstrap_auth=False))
    assert client.api_key == ""
    mint.assert_not_awaited()


def test_saved_key_still_reused_for_remote_host(isolated_credentials, quiet_client, monkeypatch):
    from cognee.api.v1.serve.credentials import CloudCredentials, save_credentials
    from cognee.api.v1.serve.serve import _serve_direct

    save_credentials(
        CloudCredentials(
            access_token="",
            service_url="https://tenant-abc.cloud.cognee.ai",
            api_key="ck_cloud",
        )
    )
    client = asyncio.run(_serve_direct("https://tenant-abc.cloud.cognee.ai"))
    assert client.api_key == "ck_cloud"
