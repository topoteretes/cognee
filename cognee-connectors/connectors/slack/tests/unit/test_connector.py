"""Contract tests for the Slack connector: ``CONNECTOR`` against Slack's real response shapes.

The token-response fixtures are the ones the moved adapter tests carry
(test_slack_adapter.py), extended with the fields a real ``oauth.v2.access`` answer
has; the Events API envelope is Slack's documented one. Nothing here touches the
network or cognee.
"""

import json
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee_connector_slack import CONNECTOR, DESCRIPTOR
from cognee_connector_slack.connector import ACCESS_URL, AUTHORIZE_URL, BOT_SCOPES
from cognee_connector_slack.settings import SlackSettings, slack_settings
from cognee_connectors_core.base import RefreshedTokens, WebhookDispatch
from cognee_connectors_core.errors import WebhookVerificationError
from cognee_connectors_core.oauth2 import DescriptorConnector
from cognee_connectors_core.registry import catalog, reset_registry, use_connector
from cognee_connectors_core.signatures import SignatureScheme
from cognee_connectors_core.testing import (
    FakeCredential,
    FakeHttpClient,
    FakeSettings,
    webhook_headers,
)

# Slack's oauth.v2.access answer for a bot install: the shape the adapter tests
# use, plus the fields Slack always sends alongside.
TOKEN_RESPONSE = {
    "ok": True,
    "access_token": "xoxb-secret",
    "token_type": "bot",
    "scope": "commands,chat:write",
    "bot_user_id": "U999",
    "app_id": "A0KRD7HC3",
    "team": {"id": "T123", "name": "Acme"},
    "enterprise": {"id": "E456", "name": "Acme Grid"},
    "authed_user": {"id": "U111"},
}

# The same app with token rotation enabled on the Slack side.
ROTATING_TOKEN_RESPONSE = {
    **TOKEN_RESPONSE,
    "refresh_token": "xoxe-refresh",
    "expires_in": 43200,
}

EVENT_ENVELOPE = {
    "token": "verification-token",
    "team_id": "T123",
    "api_app_id": "A0KRD7HC3",
    "event": {"type": "app_home_opened", "user": "U111", "channel": "D1"},
    "type": "event_callback",
    "event_id": "Ev0PV52K21",
    "event_time": 1234567890,
}


def _settings(**overrides):
    values = {
        "client_id": "slack-client-id",
        "client_secret": "slack-client-secret",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/slack/callback",
        "frontend_base_url": "http://localhost:3000",
        "env_prefix": "SLACK_",
    }
    values.update(overrides)
    return FakeSettings(**values)


def _connector(http=None, **extra):
    return DescriptorConnector(DESCRIPTOR, settings=_settings(**extra), http=http)


@pytest.fixture
def registry():
    reset_registry()
    yield
    reset_registry()


# ── the descriptor ──────────────────────────────────────────────────────────


def test_descriptor_identity_and_endpoints():
    assert DESCRIPTOR.provider == "slack"
    assert DESCRIPTOR.display_name == "Slack"
    assert DESCRIPTOR.auth == "oauth2"
    assert DESCRIPTOR.authorize_url == AUTHORIZE_URL == "https://slack.com/oauth/v2/authorize"
    assert DESCRIPTOR.token_url == ACCESS_URL == "https://slack.com/api/oauth.v2.access"
    assert DESCRIPTOR.scopes == BOT_SCOPES
    assert DESCRIPTOR.signature_scheme is SignatureScheme.SLACK_V0
    assert DESCRIPTOR.signature_header == "x-slack-signature"
    assert DESCRIPTOR.signature_timestamp_header == "x-slack-request-timestamp"


def test_descriptor_matches_the_host_oauth_module():
    # The cognee_host adapter keeps cognee's own OAuth module; the descriptor
    # must say the same thing, or the two connect paths would drift apart.
    from cognee_connector_slack.cognee_host import oauth

    assert oauth._AUTHORIZE_URL == DESCRIPTOR.authorize_url
    assert oauth._ACCESS_URL == DESCRIPTOR.token_url
    assert oauth._BOT_SCOPES == DESCRIPTOR.scopes


def test_connector_serves_the_descriptor_with_the_slack_settings():
    assert CONNECTOR.provider == "slack"
    assert CONNECTOR.display_name == "Slack"
    assert CONNECTOR.auth == "oauth2"
    assert CONNECTOR.descriptor is DESCRIPTOR
    assert CONNECTOR.settings is slack_settings
    assert isinstance(slack_settings, SlackSettings)
    assert slack_settings.model_config["env_prefix"] == "SLACK_"


# ── connect ─────────────────────────────────────────────────────────────────


def test_authorize_url_carries_client_scopes_redirect_and_state():
    url = _connector().authorize_url("state-1")

    assert url.startswith(AUTHORIZE_URL + "?")
    query = parse_qs(urlsplit(url).query)
    assert query["client_id"] == ["slack-client-id"]
    assert query["scope"] == [BOT_SCOPES]
    assert query["redirect_uri"] == ["http://localhost:8000/api/v1/integrations/slack/callback"]
    assert query["state"] == ["state-1"]


def test_authorize_url_names_the_missing_setting():
    with pytest.raises(RuntimeError, match="SLACK_CLIENT_ID is not configured"):
        _connector(client_id="").authorize_url("state-1")


@pytest.mark.asyncio
async def test_exchange_posts_the_code_to_oauth_v2_access():
    http = FakeHttpClient({("POST", ACCESS_URL): (200, TOKEN_RESPONSE)})

    body = await _connector(http).exchange_callback("code-1", {"state": "ignored"})

    assert body == TOKEN_RESPONSE
    call = http.last_call
    assert (call.method, call.url) == ("POST", ACCESS_URL)
    assert call.data == {
        "client_id": "slack-client-id",
        "client_secret": "slack-client-secret",
        "code": "code-1",
        "redirect_uri": "http://localhost:8000/api/v1/integrations/slack/callback",
        "grant_type": "authorization_code",
    }
    assert call.headers["Accept"] == "application/json"


@pytest.mark.asyncio
async def test_rejected_exchange_raises_with_slacks_error_code():
    # Slack answers HTTP 200 with ok: false; the status alone says nothing.
    http = FakeHttpClient({("POST", ACCESS_URL): (200, {"ok": False, "error": "invalid_code"})})

    with pytest.raises(RuntimeError, match="invalid_code"):
        await _connector(http).exchange_callback("code-1", {})


def test_parse_installation_reads_team_token_metadata_and_scopes():
    installation = _connector().parse_installation(TOKEN_RESPONSE)

    assert installation.provider_account_id == "T123"
    assert installation.account_label == "Acme"
    assert installation.scopes == "commands,chat:write"
    assert installation.token_payload == {"access_token": "xoxb-secret"}
    assert installation.provider_metadata == {
        "bot_user_id": "U999",
        "enterprise_id": "E456",
        "installed_by_slack_user_id": "U111",
    }
    assert installation.token_expires_at is None
    assert installation.auth_type == "oauth2"
    # Secret material never ends up in the clear-text metadata.
    assert "xoxb-secret" not in installation.provider_metadata.values()


def test_parse_installation_keeps_the_refresh_token_and_expiry_of_a_rotating_app():
    before = datetime.now(timezone.utc)

    installation = _connector().parse_installation(ROTATING_TOKEN_RESPONSE)

    assert installation.token_payload == {
        "access_token": "xoxb-secret",
        "refresh_token": "xoxe-refresh",
    }
    assert installation.token_expires_at is not None
    assert installation.token_expires_at - before >= timedelta(seconds=43199)


def test_parse_installation_refuses_a_response_without_a_team():
    # A descriptor reads team.id only; an org-level Grid install (enterprise.id,
    # no team) is the cognee_host adapter's fallback, not the descriptor's.
    with pytest.raises(RuntimeError, match="team.id"):
        _connector().parse_installation({"ok": True, "access_token": "xoxb-secret"})


# ── refresh ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_refresh_rotates_through_the_refresh_token_grant():
    http = FakeHttpClient(
        {
            ("POST", ACCESS_URL): (
                200,
                {
                    "ok": True,
                    "access_token": "xoxb-new",
                    "refresh_token": "xoxe-new",
                    "expires_in": 43200,
                    "scope": "commands,chat:write",
                },
            )
        }
    )
    credential = FakeCredential(provider="slack", provider_account_id="T123")

    refreshed = await _connector(http).refresh(
        credential, {"access_token": "xoxb-old", "refresh_token": "xoxe-old"}
    )

    assert isinstance(refreshed, RefreshedTokens)
    assert refreshed.token_payload == {"access_token": "xoxb-new", "refresh_token": "xoxe-new"}
    assert refreshed.scopes == "commands,chat:write"
    assert refreshed.token_expires_at is not None
    assert http.last_call.data == {
        "client_id": "slack-client-id",
        "client_secret": "slack-client-secret",
        "grant_type": "refresh_token",
        "refresh_token": "xoxe-old",
    }


@pytest.mark.asyncio
async def test_refresh_is_a_noop_without_a_refresh_token():
    # The normal case: token rotation is an opt-in, irreversible app setting.
    http = FakeHttpClient()
    credential = FakeCredential(provider="slack", provider_account_id="T123")

    assert await _connector(http).refresh(credential, {"access_token": "xoxb-secret"}) is None
    assert http.calls == []


@pytest.mark.asyncio
async def test_rejected_refresh_raises_with_slacks_error_code():
    http = FakeHttpClient({("POST", ACCESS_URL): (200, {"ok": False, "error": "invalid_grant"})})
    credential = FakeCredential(provider="slack", provider_account_id="T123")

    with pytest.raises(RuntimeError, match="invalid_grant"):
        await _connector(http).refresh(credential, {"refresh_token": "xoxe-old"})


# ── sync scope and dataset ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_scope_names_the_workspace_dataset_and_hands_out_the_bot_token():
    credential = FakeCredential(provider="slack", provider_account_id="T123")

    scope = await CONNECTOR.sync_scope(
        credential, {"access_token": "xoxb-secret"}, resources=["C1", "C2"]
    )

    assert scope.dataset_name == "slack_t123"
    assert scope.items == ["C1", "C2"]
    assert scope.credential == "xoxb-secret"
    assert scope.content_type is None
    assert CONNECTOR.dataset_name(credential) == "slack_t123"


@pytest.mark.asyncio
async def test_sync_scope_cannot_enumerate_a_whole_workspace():
    credential = FakeCredential(provider="slack", provider_account_id="T123")

    with pytest.raises(NotImplementedError):
        await CONNECTOR.sync_scope(credential, {"access_token": "xoxb-secret"})


# ── webhooks ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_webhook_verifier_accepts_a_slack_signed_delivery_and_refuses_the_rest():
    connector = _connector(signing_secret="s3cret")
    verifier = connector.webhook_verifier()
    body = json.dumps(EVENT_ENVELOPE).encode()
    headers = webhook_headers(DESCRIPTOR, "s3cret", body)

    assert await verifier.verify(body, headers) == body

    with pytest.raises(WebhookVerificationError) as tampered:
        await verifier.verify(body + b" ", headers)
    assert tampered.value.status_code == 401

    stale = webhook_headers(DESCRIPTOR, "s3cret", body, timestamp=int(time.time()) - 600)
    with pytest.raises(WebhookVerificationError):
        await verifier.verify(body, stale)

    with pytest.raises(WebhookVerificationError):
        await verifier.verify(body, webhook_headers(DESCRIPTOR, "other", body))


@pytest.mark.asyncio
async def test_webhook_verifier_reads_the_signing_secret_at_call_time():
    settings = _settings(signing_secret="")
    verifier = DescriptorConnector(DESCRIPTOR, settings=settings).webhook_verifier()
    body = b"{}"
    headers = webhook_headers(DESCRIPTOR, "s3cret", body)

    with pytest.raises(WebhookVerificationError):  # unset secret: never an empty-key HMAC
        await verifier.verify(body, headers)

    settings.signing_secret = "s3cret"
    assert await verifier.verify(body, headers) == body


@pytest.mark.asyncio
async def test_handle_webhook_routes_an_events_api_envelope_by_team_and_event_id():
    raw_body = json.dumps(EVENT_ENVELOPE).encode()

    dispatch = await CONNECTOR.handle_webhook(raw_body, {"content-type": "application/json"})

    assert dispatch == WebhookDispatch(
        provider_account_id="T123", resources=[], delivery_id="Ev0PV52K21"
    )
    assert CONNECTOR.webhook_delivery_key(raw_body, {}) == "Ev0PV52K21"


@pytest.mark.asyncio
async def test_handle_webhook_drops_a_form_encoded_slash_command():
    # Slash commands and interactive payloads are form-encoded and go through
    # the host's own /commands and /interactive routes.
    dispatch = await CONNECTOR.handle_webhook(b"command=%2Fcognee-ask&team_id=T123", {})

    assert dispatch.provider_account_id is None


# ── registry and catalog ────────────────────────────────────────────────────


def test_catalog_describes_slack_as_a_tier_one_webhook_connector(registry):
    use_connector(CONNECTOR)

    (entry,) = catalog()

    assert entry == {
        "provider": "slack",
        "display_name": "Slack",
        "auth": "oauth2",
        "tier": 1,
        "has_webhooks": True,
        "supports_resource_selection": False,
        "poll_interval_seconds": None,
        "content_type": None,
        "degraded": False,
    }
