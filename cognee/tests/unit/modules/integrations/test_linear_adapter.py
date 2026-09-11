"""Unit tests for cognee.modules.integrations.linear.adapter.

Network calls (code exchange, install-context GraphQL query, token revoke)
are mocked at the aiohttp/client seam — what's under test is the install
flow: the agent-install authorize URL, the identity-enriching callback, the
secret/metadata split in parse_installation, and revoke_remote's never-raise
contract.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee.modules.integrations.linear import adapter as adapter_module
from cognee.modules.integrations.linear.adapter import LinearIntegration
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
    def __init__(self, status=200, payload=None):
        self.status = status
        self._payload = payload if payload is not None else {}

    async def json(self):
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
    )
    monkeypatch.setattr(adapter_module, "aiohttp", fake)
    return session


def test_authorize_url_is_the_agent_install_flow():
    url = LinearIntegration().authorize_url("the-state")
    scheme_host_path, query = url.split("?", 1)

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


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_network_failure(monkeypatch):
    monkeypatch.setattr(adapter_module, "access_token_for", lambda _credential: "lin_oauth_tok")
    _fake_aiohttp(monkeypatch, _FakeSession(post_error=ConnectionError("network down")))

    # Best-effort by contract: a network blip must never block a disconnect.
    await LinearIntegration().revoke_remote(SimpleNamespace(provider_account_id="org-1"))


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_a_non_200_response(monkeypatch):
    monkeypatch.setattr(adapter_module, "access_token_for", lambda _credential: "lin_oauth_tok")
    session = _fake_aiohttp(monkeypatch, _FakeSession(_FakeResponse(401)))

    await LinearIntegration().revoke_remote(SimpleNamespace(provider_account_id="org-1"))

    (url, kwargs) = session.post_calls[0]
    assert url == "https://api.linear.app/oauth/revoke"
    assert kwargs["headers"] == {"Authorization": "Bearer lin_oauth_tok"}


@pytest.mark.asyncio
async def test_revoke_remote_never_raises_on_an_unusable_credential(monkeypatch):
    def _boom(_credential):
        raise RuntimeError("holds no access token")

    monkeypatch.setattr(adapter_module, "access_token_for", _boom)

    await LinearIntegration().revoke_remote(SimpleNamespace(provider_account_id="org-1"))
