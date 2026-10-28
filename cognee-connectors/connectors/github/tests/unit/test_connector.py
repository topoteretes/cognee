"""Contract tests for the GitHub connector (``cognee_connector_github.connector``).

The user-token exchange goes through the core ``HttpClient`` seam, so it is
driven with ``FakeHttpClient`` and GitHub's real token-response shapes; the
app-authenticated calls (installation lookup, token minting, repository
listing) are mocked at the ``app_auth`` seam, as the moved adapter tests do.
What is under test is the install-flow logic — the installation_id must be
verified against the completing user before it is trusted, and the stored
credential must carry no token material — the sync scope a host runs with,
and the reduction of a webhook delivery to a dispatch.
"""

import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from cognee_connector_github import CONNECTOR, app_auth
from cognee_connector_github.connector import _ACCESS_TOKEN_URL, GithubConnector
from cognee_connector_github.webhooks import (
    SIGNATURE_HEADER,
    github_webhook_verifier,
    is_valid_github_signature,
)
from cognee_connectors_core import registry
from cognee_connectors_core.base import (
    Connector,
    ConnectorInstallation,
    ConnectorResource,
    WebhookDispatch,
)
from cognee_connectors_core.errors import WebhookVerificationError
from cognee_connectors_core.signatures import HmacVerifier, SignatureScheme
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings, sign_body

# GitHub's installation record (GET /app/installations/{id}) — the fixture
# the moved adapter tests carry.
_INSTALLATION = {
    "id": 12345,
    "app_id": 7,
    "repository_selection": "selected",
    "account": {"login": "acme-org", "id": 99, "type": "Organization"},
}

# POST https://github.com/login/oauth/access_token answers HTTP 200 in both
# cases: the user token, or an error object.
_USER_TOKEN_RESPONSE = {
    "access_token": "ghu_16C7e42F292c6912E7710c838347Ae178B4a",
    "token_type": "bearer",
    "scope": "",
}
_BAD_CODE_RESPONSE = {
    "error": "bad_verification_code",
    "error_description": "The code passed is incorrect or expired.",
    "error_uri": (
        "https://docs.github.com/apps/managing-oauth-apps/"
        "troubleshooting-oauth-app-access-token-request-errors/#bad-verification-code"
    ),
}

# GET /installation/repositories records, reduced to the fields read here.
_REPOSITORIES = [
    {
        "id": 1296269,
        "full_name": "acme/api",
        "description": "The API",
        "private": True,
        "archived": False,
        "default_branch": "main",
        "language": "Python",
    },
    {
        "id": 1296270,
        "full_name": "acme/web",
        "description": None,
        "private": False,
        "archived": True,
        "default_branch": "trunk",
        "language": "TypeScript",
    },
]

_EXPIRES_AT = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
_SECRET = "hook-secret"


def _settings(**overrides) -> FakeSettings:
    values = {
        "env_prefix": "GITHUB_",
        "client_id": "Iv1.abc",
        "client_secret": "shhh",
        "app_slug": "cognee-app",
        "webhook_secret": _SECRET,
        "frontend_base_url": "https://app.example.com",
    }
    values.update(overrides)
    return FakeSettings(**values)


def _connector(http: FakeHttpClient | None = None, **overrides) -> GithubConnector:
    return GithubConnector(settings=_settings(**overrides), http=http or FakeHttpClient())


def _credential(**overrides) -> FakeCredential:
    defaults = {
        "provider": "github",
        "provider_account_id": "12345",
        "account_label": "acme-org",
        "provider_metadata": {"account_login": "acme-org"},
        "auth_type": "github_app",
    }
    return FakeCredential(**{**defaults, **overrides})


@pytest.fixture
def app_calls():
    """``app_auth`` mocked at the seam the connector calls it through."""
    with (
        patch.object(
            app_auth, "mint_installation_token", new=AsyncMock(return_value=("tok123", _EXPIRES_AT))
        ) as mint,
        patch.object(
            app_auth,
            "list_installation_repository_records",
            new=AsyncMock(return_value=list(_REPOSITORIES)),
        ) as list_records,
        patch.object(
            app_auth, "user_can_access_installation", new=AsyncMock(return_value=True)
        ) as access_check,
        patch.object(
            app_auth, "get_installation", new=AsyncMock(return_value=_INSTALLATION)
        ) as lookup,
    ):
        yield {
            "mint": mint,
            "list_records": list_records,
            "access_check": access_check,
            "lookup": lookup,
        }


# ---- Identity --------------------------------------------------------------


def test_connector_identity():
    assert isinstance(CONNECTOR, Connector)
    assert isinstance(CONNECTOR, GithubConnector)
    assert CONNECTOR.provider == "github"
    assert CONNECTOR.display_name == "GitHub"
    assert CONNECTOR.auth == "oauth2"
    assert CONNECTOR.content_type == "code"
    # Webhook-driven: nothing to poll.
    assert CONNECTOR.poll_interval_seconds is None
    assert CONNECTOR.poll_targets(_credential()) is None
    assert GithubConnector.supports_resource_selection() is True
    assert registry.tier_of(CONNECTOR) == 3
    assert repr(CONNECTOR) == "<GithubConnector provider='github'>"


def test_catalog_entry():
    registry.reset_registry()
    try:
        registry.use_connector(CONNECTOR)
        (entry,) = registry.catalog()
    finally:
        registry.reset_registry()

    assert entry == {
        "provider": "github",
        "display_name": "GitHub",
        "auth": "oauth2",
        "tier": 3,
        "has_webhooks": True,
        "supports_resource_selection": True,
        "poll_interval_seconds": None,
        "content_type": "code",
        "degraded": False,
    }


# ---- Connect ---------------------------------------------------------------


def test_authorize_url_is_the_app_installation_flow():
    url = _connector().authorize_url("the-state")
    assert url == "https://github.com/apps/cognee-app/installations/new?state=the-state"


def test_authorize_url_carries_the_state_verbatim_but_encoded():
    url = _connector().authorize_url("a:b c")
    assert url.endswith("?state=a%3Ab+c")


def test_authorize_url_needs_the_app_slug():
    with pytest.raises(RuntimeError, match="GITHUB_APP_SLUG is not configured"):
        _connector(app_slug="").authorize_url("the-state")


def test_settings_are_read_at_call_time():
    settings = _settings(app_slug="first")
    connector = GithubConnector(settings=settings, http=FakeHttpClient())
    assert "apps/first/" in connector.authorize_url("s")
    settings.app_slug = "second"
    assert "apps/second/" in connector.authorize_url("s")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {},
        {"installation_id": ""},
        {"installation_id": "not-a-number"},
        {"installation_id": "12 345"},
        # Arabic-Indic digits: str.isdigit() is true, int() parses a different number.
        {"installation_id": "١٢٣"},
        {"installation_id": "²"},  # superscript two
    ],
)
async def test_exchange_callback_rejects_a_missing_installation_id(params):
    http = FakeHttpClient()
    with pytest.raises(ValueError, match="installation_id"):
        await _connector(http).exchange_callback("code", params)
    # Refused before any provider call.
    assert http.calls == []


@pytest.mark.asyncio
async def test_exchange_callback_rejects_an_implausibly_long_installation_id():
    http = FakeHttpClient()
    with pytest.raises(ValueError, match="implausibly long"):
        await _connector(http).exchange_callback("code", {"installation_id": "1" * 21})
    assert http.calls == []


@pytest.mark.asyncio
async def test_exchange_callback_verifies_user_access_before_trusting_the_id(app_calls):
    http = FakeHttpClient({("POST", _ACCESS_TOKEN_URL): (200, _USER_TOKEN_RESPONSE)})

    result = await _connector(http).exchange_callback("the-code", {"installation_id": "12345"})

    assert result == _INSTALLATION
    # The code exchange is a plain OAuth form post, through the HttpClient seam.
    (call,) = http.calls
    assert call.method == "POST"
    assert call.url == _ACCESS_TOKEN_URL
    assert call.data == {"client_id": "Iv1.abc", "client_secret": "shhh", "code": "the-code"}
    assert call.headers["Accept"] == "application/json"
    app_calls["access_check"].assert_awaited_once_with(
        "ghu_16C7e42F292c6912E7710c838347Ae178B4a", 12345
    )
    # The credential comes from the app-authenticated lookup, never from the
    # unauthenticated callback query string.
    app_calls["lookup"].assert_awaited_once_with(12345)


@pytest.mark.asyncio
async def test_exchange_callback_refuses_an_installation_the_user_cannot_access(app_calls):
    app_calls["access_check"].return_value = False
    http = FakeHttpClient({("POST", _ACCESS_TOKEN_URL): (200, _USER_TOKEN_RESPONSE)})

    with pytest.raises(PermissionError):
        await _connector(http).exchange_callback("the-code", {"installation_id": "12345"})

    app_calls["lookup"].assert_not_awaited()


@pytest.mark.asyncio
async def test_exchange_callback_reports_githubs_error_body(app_calls):
    # GitHub answers a bad code with HTTP 200 and an error object.
    http = FakeHttpClient({("POST", _ACCESS_TOKEN_URL): (200, _BAD_CODE_RESPONSE)})

    with pytest.raises(RuntimeError, match="bad_verification_code"):
        await _connector(http).exchange_callback("stale", {"installation_id": "12345"})

    app_calls["access_check"].assert_not_awaited()
    app_calls["lookup"].assert_not_awaited()


@pytest.mark.asyncio
async def test_exchange_callback_reports_a_response_without_a_token(app_calls):
    http = FakeHttpClient({("POST", _ACCESS_TOKEN_URL): (200, {"token_type": "bearer"})})

    with pytest.raises(RuntimeError, match="no access_token"):
        await _connector(http).exchange_callback("code", {"installation_id": "12345"})


@pytest.mark.asyncio
async def test_exchange_callback_reports_a_non_json_body(app_calls):
    http = FakeHttpClient({("POST", _ACCESS_TOKEN_URL): (502, "<html>Bad gateway</html>")})

    with pytest.raises(RuntimeError, match="non-JSON body \\(HTTP 502\\)"):
        await _connector(http).exchange_callback("code", {"installation_id": "12345"})


@pytest.mark.asyncio
async def test_exchange_callback_needs_the_oauth_app_credentials():
    http = FakeHttpClient()
    with pytest.raises(RuntimeError, match="GITHUB_CLIENT_SECRET is not configured"):
        await _connector(http, client_secret="").exchange_callback(
            "code", {"installation_id": "12345"}
        )
    assert http.calls == []


def test_parse_installation_stores_no_token_material():
    installation = _connector().parse_installation(_INSTALLATION)

    assert isinstance(installation, ConnectorInstallation)
    assert installation.provider_account_id == "12345"
    # No durable per-installation secret exists — tokens are minted on
    # demand from the app private key, so the encrypted payload stays empty.
    assert installation.token_payload == {}
    assert installation.auth_type == "github_app"
    assert installation.account_label == "acme-org"
    assert installation.scopes is None
    assert installation.token_expires_at is None
    assert installation.provider_metadata == {
        "account_login": "acme-org",
        "account_id": 99,
        "account_type": "Organization",
        "repository_selection": "selected",
        "app_id": 7,
    }


def test_parse_installation_without_id_raises():
    with pytest.raises(RuntimeError, match="no id"):
        _connector().parse_installation({"account": {"login": "acme-org"}})


def test_parse_installation_tolerates_a_missing_account():
    installation = _connector().parse_installation({"id": 7, "account": None})
    assert installation.provider_account_id == "7"
    assert installation.account_label is None
    assert installation.provider_metadata["account_login"] is None


# ---- Sync ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_scope_covers_every_installation_repo_by_default(app_calls):
    scope = await _connector().sync_scope(_credential(), {})

    app_calls["mint"].assert_awaited_once_with(12345)
    app_calls["list_records"].assert_awaited_once_with("tok123")
    # One dataset per installation, keyed on the installation id.
    assert scope.dataset_name == "github_12345"
    assert scope.content_type == "code"
    assert scope.items == ["https://github.com/acme/api.git", "https://github.com/acme/web.git"]
    assert scope.item_resource_ids == ["acme/api", "acme/web"]
    assert scope.credential == "tok123"
    assert scope.credential_expires_at == _EXPIRES_AT
    assert scope.credential_refreshable is True
    assert scope.options == {}


@pytest.mark.asyncio
async def test_sync_scope_with_resources_skips_the_listing_call(app_calls):
    scope = await _connector().sync_scope(_credential(), {}, resources=["acme/api"])

    app_calls["list_records"].assert_not_awaited()
    assert scope.items == ["https://github.com/acme/api.git"]
    assert scope.item_resource_ids == ["acme/api"]
    assert scope.credential == "tok123"


@pytest.mark.asyncio
async def test_sync_scope_refuses_an_empty_resource_list(app_calls):
    with pytest.raises(ValueError, match="empty"):
        await _connector().sync_scope(_credential(), {}, resources=[])
    app_calls["mint"].assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_scope_drops_resources_that_are_not_full_names(app_calls):
    scope = await _connector().sync_scope(
        _credential(), {}, resources=["../evil", "acme", "acme/api", "a/b/c", ""]
    )
    assert scope.items == ["https://github.com/acme/api.git"]
    assert scope.item_resource_ids == ["acme/api"]


@pytest.mark.asyncio
async def test_sync_scope_mints_rather_than_reading_stored_tokens(app_calls):
    # The stored payload is empty by design; whatever a host passes is not a
    # GitHub credential and must never reach the sync runner.
    scope = await _connector().sync_scope(_credential(), {"access_token": "stale"})
    assert scope.credential == "tok123"


@pytest.mark.asyncio
async def test_clone_urls_carry_no_credentials(app_calls):
    scope = await _connector().sync_scope(_credential(), {})
    assert all("tok123" not in item for item in scope.items)


@pytest.mark.asyncio
async def test_list_resources_lists_the_installations_repositories(app_calls):
    resources = await _connector().list_resources(_credential(), {})

    app_calls["mint"].assert_awaited_once_with(12345)
    app_calls["list_records"].assert_awaited_once_with("tok123")
    assert resources == [
        ConnectorResource(
            id="acme/api",
            name="acme/api",
            description="The API",
            attributes={
                "id": 1296269,
                "private": True,
                "archived": False,
                "default_branch": "main",
                "language": "Python",
            },
        ),
        ConnectorResource(
            id="acme/web",
            name="acme/web",
            description=None,
            attributes={
                "id": 1296270,
                "private": False,
                "archived": True,
                "default_branch": "trunk",
                "language": "TypeScript",
            },
        ),
    ]
    # The picker's ids are the identity sync_scope filters on.
    scope = await _connector().sync_scope(_credential(), {}, resources=[r.id for r in resources])
    assert scope.item_resource_ids == [r.id for r in resources]


def test_dataset_name_is_keyed_on_the_installation_id():
    assert _connector().dataset_name(_credential()) == "github_12345"
    # Never on the label: an org rename must not move the dataset.
    assert _connector().dataset_name(_credential(account_label="renamed-org")) == "github_12345"


@pytest.mark.asyncio
async def test_nothing_to_refresh_or_revoke_remotely():
    assert await _connector().refresh(_credential(), {}) is None
    assert await _connector().revoke_remote(_credential(), {}) is None


# ---- Webhooks --------------------------------------------------------------


def _signed(body: bytes, secret: str = _SECRET) -> dict[str, str]:
    return {"X-Hub-Signature-256": sign_body(SignatureScheme.SHA256_PREFIX, secret, body)}


def test_webhook_verifier_is_the_core_sha256_prefix_scheme():
    verifier = _connector().webhook_verifier()
    assert isinstance(verifier, HmacVerifier)
    assert verifier.scheme is SignatureScheme.SHA256_PREFIX
    assert verifier.header == SIGNATURE_HEADER == "x-hub-signature-256"
    assert verifier.timestamp_header is None


@pytest.mark.asyncio
async def test_webhook_verifier_accepts_a_signed_delivery_in_any_header_case():
    body = b'{"action":"created","installation":{"id":42}}'
    verifier = _connector().webhook_verifier()

    assert await verifier.verify(body, _signed(body)) == body
    assert await verifier.verify(
        body, {"x-hub-signature-256": _signed(body)["X-Hub-Signature-256"]}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Hub-Signature-256": ""},
        {"X-Hub-Signature-256": "sha256=" + "0" * 64},
        {"X-Hub-Signature-256": sign_body(SignatureScheme.BARE_HEX, _SECRET, b'{"a":1}')},
        # sha1= is GitHub's legacy header, deliberately unsupported.
        {"X-Hub-Signature": "sha1=" + "0" * 40},
    ],
)
async def test_webhook_verifier_rejects_unsigned_and_forged_deliveries(headers):
    with pytest.raises(WebhookVerificationError) as info:
        await _connector().webhook_verifier().verify(b'{"a":1}', headers)
    assert info.value.status_code == 401


@pytest.mark.asyncio
async def test_webhook_verifier_rejects_a_tampered_body():
    signed = _signed(b'{"action":"created"}')
    with pytest.raises(WebhookVerificationError):
        await _connector().webhook_verifier().verify(b'{"action":"deleted"}', signed)


@pytest.mark.asyncio
async def test_webhook_verifier_refuses_everything_without_a_secret():
    body = b"payload"
    with pytest.raises(WebhookVerificationError):
        await _connector(webhook_secret="").webhook_verifier().verify(body, _signed(body, ""))


@pytest.mark.asyncio
async def test_webhook_verifier_reads_the_secret_per_delivery():
    settings = _settings(webhook_secret="old")
    verifier = GithubConnector(settings=settings, http=FakeHttpClient()).webhook_verifier()
    body = b"payload"
    assert await verifier.verify(body, _signed(body, "old")) == body
    settings.webhook_secret = "new"
    assert await verifier.verify(body, _signed(body, "new")) == body
    with pytest.raises(WebhookVerificationError):
        await verifier.verify(body, _signed(body, "old"))


@pytest.mark.asyncio
async def test_core_verifier_agrees_with_the_reference_signature_check(monkeypatch):
    monkeypatch.setattr("cognee_connector_github.settings.github_settings.webhook_secret", _SECRET)
    body = b'{"action":"created","installation":{"id":42}}'
    signature = _signed(body)["X-Hub-Signature-256"]

    assert is_valid_github_signature(body, signature)
    assert await github_webhook_verifier().verify(body, {SIGNATURE_HEADER: signature}) == body

    forged = "sha256=" + "f" * 64
    assert not is_valid_github_signature(body, forged)
    with pytest.raises(WebhookVerificationError):
        await github_webhook_verifier().verify(body, {SIGNATURE_HEADER: forged})


def test_webhook_delivery_key_is_the_body_digest():
    body = b'{"action":"created","installation":{"id":42}}'
    key = _connector().webhook_delivery_key(body, {"X-GitHub-Delivery": "ignored"})
    assert key == "body-sha256:" + hashlib.sha256(body).hexdigest()
    # A header is not part of the HMAC input, so it never influences the key.
    assert key == _connector().webhook_delivery_key(body, {"X-GitHub-Delivery": "other"})
    assert key != _connector().webhook_delivery_key(body + b" ", {})


def _body(payload: dict) -> bytes:
    import json

    return json.dumps(payload).encode()


@pytest.mark.asyncio
async def test_installation_deleted_revokes(app_calls):
    body = _body({"action": "deleted", "installation": {"id": 42}})
    dispatch = await _connector().handle_webhook(body, {"X-GitHub-Event": "installation"})

    assert dispatch == WebhookDispatch(
        provider_account_id="42",
        revoke=True,
        delivery_id=_connector().webhook_delivery_key(body, {}),
    )
    app_calls["mint"].assert_not_awaited()


@pytest.mark.asyncio
async def test_installation_suspend_is_a_suspend_not_a_revoke(app_calls):
    body = _body({"action": "suspend", "installation": {"id": 42}})
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "installation"})

    assert dispatch.provider_account_id == "42"
    assert dispatch.suspend is True
    assert dispatch.revoke is False
    assert dispatch.resources == []


@pytest.mark.asyncio
async def test_installation_unsuspend_resyncs_every_repository(app_calls):
    body = _body({"action": "unsuspend", "installation": {"id": 42}})
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "installation"})

    app_calls["mint"].assert_awaited_once_with(42)
    app_calls["list_records"].assert_awaited_once_with("tok123")
    assert dispatch.provider_account_id == "42"
    assert dispatch.resources == ["acme/api", "acme/web"]
    assert dispatch.revoke is False and dispatch.suspend is False


@pytest.mark.asyncio
async def test_added_repositories_are_synced_removed_only_logged(app_calls):
    body = _body(
        {
            "action": "added",
            "installation": {"id": 42},
            "repositories_added": [{"id": 1, "full_name": "acme/new"}, None, {"id": 2}],
            "repositories_removed": [{"id": 3, "full_name": "acme/old"}],
        }
    )
    dispatch = await _connector().handle_webhook(
        body, {"x-github-event": "installation_repositories"}
    )

    # Removed repos never trigger deletion — indexed data outlives the
    # webhook; forget() stays a human decision.
    assert dispatch.resources == ["acme/new"]
    assert dispatch.provider_account_id == "42"


@pytest.mark.asyncio
async def test_push_to_default_branch_resyncs_that_repo(app_calls):
    body = _body(
        {
            "ref": "refs/heads/main",
            "installation": {"id": 42},
            "repository": {"id": 1, "full_name": "acme/api", "default_branch": "main"},
        }
    )
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "push"})
    assert dispatch.resources == ["acme/api"]
    assert dispatch.provider_account_id == "42"


@pytest.mark.asyncio
async def test_push_to_another_branch_implies_no_work(app_calls):
    body = _body(
        {
            "ref": "refs/heads/feature-x",
            "installation": {"id": 42},
            "repository": {"id": 1, "full_name": "acme/api", "default_branch": "main"},
        }
    )
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "push"})
    # Understood, nothing to sync: an account but no resources, never None.
    assert dispatch.provider_account_id == "42"
    assert dispatch.resources == []


@pytest.mark.asyncio
async def test_other_events_name_the_account_only(app_calls):
    body = _body({"action": "created", "installation": {"id": 42}})
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "issues"})
    assert dispatch.provider_account_id == "42"
    assert dispatch.resources == []
    assert dispatch.delivery_id == _connector().webhook_delivery_key(body, {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff\xfe",
        b"[1, 2]",
        b'"a string"',
        _body({"zen": "Design for failure."}),
        _body({"installation": "42"}),
        _body({"installation": {"id": None}}),
    ],
)
async def test_deliveries_without_an_installation_are_dropped(app_calls, body):
    dispatch = await _connector().handle_webhook(body, {"x-github-event": "ping"})
    assert dispatch == WebhookDispatch(provider_account_id=None)
    app_calls["mint"].assert_not_awaited()
