"""Unit tests for cognee.modules.integrations.linear.adapter.

Network calls (code exchange, install-context GraphQL query, token refresh and
revoke) are mocked at the aiohttp/client seam. What's under test is the install
flow: the agent-install authorize URL, the identity-enriching callback, the
secret/metadata split in parse_installation, the refresh request and its error
mapping, and revoke_remote's never-raise contract. Refresh against a real
credential row lives in test_linear_credential_lifecycle.py.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import aiohttp
import pytest

from cognee.modules.integrations.linear import adapter as adapter_module
from cognee.modules.integrations.linear.adapter import LinearAuthError, LinearIntegration
from cognee.modules.integrations.linear.verify_linear_signature import LinearWebhookVerifier

_TOKEN_RESPONSE = {
    "access_token": "lin_oauth_tok",
    "refresh_token": "lin_refresh_tok",
    "token_type": "Bearer",
    "expires_in": 86400,
    "scope": "read,write,app:assignable,app:mentionable",
}

_INSTALL_CONTEXT = {
    "viewer": {"id": "app-user-1", "name": "cognee-agent"},
    "organization": {"id": "org-1", "name": "Acme Co", "urlKey": "acme-co"},
}


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = "cognee.modules.integrations.linear.linear_settings.linear_settings"
    monkeypatch.setattr(f"{settings}.client_id", "lin_client")
    monkeypatch.setattr(f"{settings}.client_secret", "shhh")
    monkeypatch.setattr(
        f"{settings}.redirect_uri", "http://localhost:8000/api/v1/integrations/linear/callback"
    )
    monkeypatch.setattr(f"{settings}.webhook_secret", "hook-secret")
    monkeypatch.setattr(f"{settings}.frontend_base_url", "http://localhost:3000")


class _FakeResponse:
    def __init__(self, status=200, payload=None, json_error=None):
        self.status = status
        self._payload = payload if payload is not None else {}
        self._json_error = json_error

    async def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeSession:
    def __init__(self, response=None, post_error=None):
        self._response = response
        self._post_error = post_error
        self.post_calls = []

    def post(self, url, **kwargs):
        if self._post_error is not None:
            raise self._post_error
        self.post_calls.append((url, kwargs))
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _fake_aiohttp(monkeypatch, session):
    fake = SimpleNamespace(
        ClientSession=lambda **_kwargs: session,
        ClientTimeout=lambda **_kwargs: None,
        ClientError=aiohttp.ClientError,
    )
    monkeypatch.setattr(adapter_module, "aiohttp", fake)
    return session


def test_authorize_url_is_the_agent_install_flow():
    url = LinearIntegration().authorize_url("the-state")
    scheme_host_path, _query = url.split("?", 1)

    assert scheme_host_path == "https://linear.app/oauth/authorize"
    params = parse_qs(urlsplit(url).query)
    assert params["client_id"] == ["lin_client"]
    assert params["response_type"] == ["code"]
    assert params["state"] == ["the-state"]
    # actor=app is what makes this an agent install (an app user in the
    # workspace) rather than acting as the authorizing human.
    assert params["actor"] == ["app"]
    assert params["scope"] == ["read,write,app:assignable,app:mentionable"]


def test_state_signing_secret_is_the_webhook_secret():
    assert LinearIntegration().state_signing_secret() == "hook-secret"


def test_webhook_verifier_is_registered():
    assert isinstance(LinearIntegration().webhook_verifier(), LinearWebhookVerifier)


@pytest.mark.asyncio
async def test_exchange_code_rejects_a_response_without_access_token(monkeypatch):
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200, {"token_type": "Bearer"})))

    with pytest.raises(RuntimeError, match="no access_token"):
        await LinearIntegration().exchange_code("the-code")


@pytest.mark.asyncio
async def test_exchange_code_rejects_a_non_200_response(monkeypatch):
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(500)))

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await LinearIntegration().exchange_code("the-code")


@pytest.mark.asyncio
async def test_exchange_callback_merges_workspace_identity_into_the_token_response():
    integration = LinearIntegration()
    with (
        patch.object(
            integration, "exchange_code", new=AsyncMock(return_value=dict(_TOKEN_RESPONSE))
        ),
        patch.object(
            adapter_module, "graphql", new=AsyncMock(return_value=_INSTALL_CONTEXT)
        ) as graphql,
    ):
        result = await integration.exchange_callback("the-code", {})

    # The fresh token is spent on the install-context query so the sync
    # parse_installation can key the credential on the organization id.
    assert graphql.await_args.args[0] == "lin_oauth_tok"
    assert result["access_token"] == "lin_oauth_tok"
    assert result["viewer"] == _INSTALL_CONTEXT["viewer"]
    assert result["organization"] == _INSTALL_CONTEXT["organization"]


def test_parse_installation_splits_secrets_from_metadata():
    installation = LinearIntegration().parse_installation({**_TOKEN_RESPONSE, **_INSTALL_CONTEXT})

    assert installation.provider_account_id == "org-1"
    # Token material lives ONLY in the encrypted payload; the queryable
    # metadata carries identity, never secrets.
    assert installation.token_payload == {
        "access_token": "lin_oauth_tok",
        "refresh_token": "lin_refresh_tok",
    }
    assert installation.provider_metadata == {
        "app_user_id": "app-user-1",
        "app_user_name": "cognee-agent",
        "organization_name": "Acme Co",
        "organization_url_key": "acme-co",
        "scope": "read,write,app:assignable,app:mentionable",
        # a (re)install starts without the previous install's sync markers
        "dlt_seeded": False,
        "legacy_cleaned": False,
        "resume_needed_at": None,
    }
    assert installation.account_label == "Acme Co"
    assert installation.auth_type == "oauth2"
    assert installation.token_expires_at is not None


def test_parse_installation_without_refresh_token_stores_only_the_access_token():
    token_response = {**_TOKEN_RESPONSE, **_INSTALL_CONTEXT}
    del token_response["refresh_token"]

    installation = LinearIntegration().parse_installation(token_response)

    assert installation.token_payload == {"access_token": "lin_oauth_tok"}


def test_parse_installation_without_organization_id_raises():
    with pytest.raises(ValueError, match="organization"):
        LinearIntegration().parse_installation(dict(_TOKEN_RESPONSE))
    with pytest.raises(ValueError, match="organization"):
        LinearIntegration().parse_installation(
            {**_TOKEN_RESPONSE, "organization": {"name": "Acme Co"}}
        )


def _stored_tokens(monkeypatch, payload):
    monkeypatch.setattr(adapter_module, "decrypt_token_payload", lambda _credential: payload)


_CREDENTIAL = SimpleNamespace(provider_account_id="org-1")


@pytest.mark.asyncio
async def test_revoke_remote_revokes_the_refresh_token_in_the_token_field(monkeypatch):
    _stored_tokens(monkeypatch, {"access_token": "lin_access", "refresh_token": "lin_refresh"})
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200)))

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    (url, kwargs) = session.post_calls[0]
    assert url == "https://api.linear.app/oauth/revoke"
    assert kwargs["data"] == {"token": "lin_refresh", "token_type_hint": "refresh_token"}
    # The Authorization header still works, but only for backwards compatibility.
    assert "headers" not in kwargs


@pytest.mark.asyncio
async def test_revoke_remote_falls_back_to_the_access_token(monkeypatch):
    _stored_tokens(monkeypatch, {"access_token": "lin_access"})
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200)))

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    assert session.post_calls[0][1]["data"] == {
        "token": "lin_access",
        "token_type_hint": "access_token",
    }


@pytest.mark.asyncio
async def test_revoke_remote_sends_nothing_when_no_token_is_stored(monkeypatch):
    _stored_tokens(monkeypatch, {})
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200)))

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    assert session.post_calls == []


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_network_failure(monkeypatch):
    _stored_tokens(monkeypatch, {"refresh_token": "lin_refresh"})
    _fake_aiohttp(monkeypatch, _FakeSession(post_error=ConnectionError("network down")))

    # Best-effort by contract: a network blip must never block a disconnect.
    await LinearIntegration().revoke_remote(_CREDENTIAL)


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_a_non_200_response(monkeypatch):
    _stored_tokens(monkeypatch, {"refresh_token": "lin_refresh"})
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(400)))

    await LinearIntegration().revoke_remote(_CREDENTIAL)


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_an_unusable_credential(monkeypatch):
    def _boom(_credential):
        raise RuntimeError("cannot decrypt")

    monkeypatch.setattr(adapter_module, "decrypt_token_payload", _boom)

    await LinearIntegration().revoke_remote(_CREDENTIAL)


def test_refresh_stays_inside_linears_ten_second_window():
    # A refresh runs before the first agent activity, which Linear wants
    # within 10 seconds of a session being created.
    assert adapter_module._REFRESH_TIMEOUT.total < 10


@pytest.mark.asyncio
async def test_refresh_access_token_posts_a_form_encoded_refresh_grant(monkeypatch):
    rotated = {
        "access_token": "new_access",
        "refresh_token": "new_refresh",
        "expires_in": 86399,
        "scope": "read write",
    }
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200, rotated)))

    result = await adapter_module.refresh_access_token(
        "old_refresh", client_id="lin_client", client_secret="shhh"
    )

    assert result == rotated
    (url, kwargs) = session.post_calls[0]
    assert url == "https://api.linear.app/oauth/token"
    assert kwargs["data"] == {
        "refresh_token": "old_refresh",
        "client_id": "lin_client",
        "client_secret": "shhh",
        "grant_type": "refresh_token",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "code"),
    [
        (
            _FakeResponse(400, {"error": "invalid_grant", "error_description": "secret"}),
            "invalid_grant",
        ),
        (_FakeResponse(401, {"error": "invalid_client"}), "invalid_client"),
        (_FakeResponse(500, {}), "http_500"),
        (_FakeResponse(200, {"token_type": "Bearer"}), "no_access_token"),
    ],
)
async def test_refresh_access_token_reduces_a_failure_to_a_stable_code(monkeypatch, response, code):
    _fake_aiohttp(monkeypatch, _FakeSession(response))

    with pytest.raises(LinearAuthError) as caught:
        await adapter_module.refresh_access_token(
            "old_refresh", client_id="lin_client", client_secret="shhh"
        )

    assert caught.value.code == code
    assert str(caught.value) == f"Linear token refresh failed: {code}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ValueError("secret response body"), aiohttp.ClientPayloadError("secret body")]
)
async def test_refresh_access_token_hides_an_unreadable_response_body(monkeypatch, error):
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(502, json_error=error)))

    with pytest.raises(LinearAuthError) as caught:
        await adapter_module.refresh_access_token(
            "old_refresh", client_id="lin_client", client_secret="shhh"
        )

    assert caught.value.code == "http_502"
    assert caught.value.__suppress_context__ is True
    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_refresh_access_token_uses_the_refresh_timeout(monkeypatch):
    session = _FakeSession(_FakeResponse(200, {"access_token": "new_access"}))
    opened = []

    def client_session(**kwargs):
        opened.append(kwargs)
        return session

    monkeypatch.setattr(
        adapter_module,
        "aiohttp",
        SimpleNamespace(ClientSession=client_session, ClientError=aiohttp.ClientError),
    )

    await adapter_module.refresh_access_token("old_refresh", client_id="id", client_secret="s")

    assert opened == [{"timeout": adapter_module._REFRESH_TIMEOUT}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "proxy echoed client_secret=SEKRIT"}},
        {"error": "free text with SEKRIT and spaces"},
        None,
        [],
    ],
)
async def test_refresh_access_token_never_puts_response_text_in_the_error(monkeypatch, body):
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(400, body)))

    with pytest.raises(LinearAuthError) as caught:
        await adapter_module.refresh_access_token("old", client_id="id", client_secret="s")

    assert caught.value.code == "http_400"
    assert "SEKRIT" not in str(caught.value)


@pytest.mark.asyncio
async def test_revoke_remote_revokes_the_access_token_too(monkeypatch):
    _stored_tokens(monkeypatch, {"access_token": "lin_access", "refresh_token": "lin_refresh"})
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200)))

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    assert [call[1]["data"]["token_type_hint"] for call in session.post_calls] == [
        "refresh_token",
        "access_token",
    ]


class _FirstPostFails:
    def __init__(self, first):
        self.first = first
        self.post_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append(kwargs["data"]["token_type_hint"])
        if len(self.post_calls) == 1:
            if isinstance(self.first, Exception):
                raise self.first
            return _FakeResponse(self.first)
        return _FakeResponse(200)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [ConnectionError("network down"), 400])
async def test_a_failed_refresh_token_revoke_still_revokes_the_access_token(monkeypatch, first):
    _stored_tokens(monkeypatch, {"access_token": "lin_access", "refresh_token": "lin_refresh"})
    session = _fake_aiohttp(monkeypatch, _FirstPostFails(first))

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    assert session.post_calls == ["refresh_token", "access_token"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["lin_oauth_SECRETSECRET", "a" * 65, "invalid_grant\n"])
async def test_refresh_access_token_reports_only_known_error_codes(monkeypatch, error):
    _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(400, {"error": error})))

    with pytest.raises(LinearAuthError) as caught:
        await adapter_module.refresh_access_token("old", client_id="id", client_secret="s")

    assert caught.value.code == "http_400"


def test_revoke_runs_under_a_timeout_short_enough_for_two_in_a_row():
    assert 2 * adapter_module._REVOKE_TIMEOUT.total < adapter_module._TIMEOUT.total


def test_a_token_without_a_refresh_token_keeps_its_own_long_expiry():
    token_response = {**_TOKEN_RESPONSE, **_INSTALL_CONTEXT, "expires_in": 315359999}
    del token_response["refresh_token"]

    installation = LinearIntegration().parse_installation(token_response)

    remaining = installation.token_expires_at - datetime.now(timezone.utc)
    assert remaining > timedelta(days=3000)


@pytest.mark.asyncio
async def test_every_oauth_post_refuses_redirects(monkeypatch):
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(200, {"access_token": "a"})))
    _stored_tokens(monkeypatch, {"access_token": "lin_access", "refresh_token": "lin_refresh"})

    await adapter_module.refresh_access_token("old", client_id="id", client_secret="s")
    await LinearIntegration().revoke_remote(_CREDENTIAL)
    await LinearIntegration().exchange_code("the-code")

    assert len(session.post_calls) == 4
    assert all(kwargs["allow_redirects"] is False for _url, kwargs in session.post_calls)


@pytest.mark.asyncio
async def test_revoke_remote_opens_its_sessions_with_the_revoke_timeout(monkeypatch):
    session = _FakeSession(_FakeResponse(200))
    _stored_tokens(monkeypatch, {"access_token": "lin_access", "refresh_token": "lin_refresh"})
    opened = []

    def client_session(**kwargs):
        opened.append(kwargs)
        return session

    monkeypatch.setattr(
        adapter_module,
        "aiohttp",
        SimpleNamespace(ClientSession=client_session, ClientError=aiohttp.ClientError),
    )

    await LinearIntegration().revoke_remote(_CREDENTIAL)

    assert opened == [{"timeout": adapter_module._REVOKE_TIMEOUT}] * 2


@pytest.mark.parametrize("expires_in", [None, 0, "abc", 10**30, True])
def test_parse_installation_gives_an_unusable_expires_in_a_day(expires_in):
    installation = LinearIntegration().parse_installation(
        {**_TOKEN_RESPONSE, **_INSTALL_CONTEXT, "expires_in": expires_in}
    )

    remaining = installation.token_expires_at - datetime.now(timezone.utc)
    assert timedelta(hours=23, minutes=59) < remaining <= timedelta(hours=24)
