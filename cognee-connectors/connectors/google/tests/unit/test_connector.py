"""Contract tests for the shared Google connector: ``GoogleConnector`` against Google's shapes.

The token response and userinfo fixtures are the ones cognee's adapter tests carry
(``test_google_oauth_adapters.py``, ``test_google_drive_adapter.py``); Google's token and
userinfo endpoints are faked with ``FakeHttpClient``, so nothing here touches the network,
and nothing imports cognee.
"""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee_connector_google import CONNECTOR, client
from cognee_connector_google.connector import (
    AUTHORIZE_URL,
    DESCRIPTOR,
    IDENTITY_SCOPES,
    POLL_INTERVAL_SECONDS,
    GoogleConnector,
    google_descriptor,
)
from cognee_connectors_core import registry
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings

SUBJECT = "110000000000000000001"
SCOPE = "openid email https://www.googleapis.com/auth/drive.readonly"
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": SCOPE,
    "token_type": "Bearer",
}
WORKSPACE_USERINFO = {"sub": SUBJECT, "email": "goran@topoteretes.com", "hd": "topoteretes.com"}
PERSONAL_USERINFO = {"sub": "110000000000000000002", "email": "someone@gmail.com"}
# Google issues no new refresh token on a refresh.
REFRESH_RESPONSE = {
    "access_token": "ya29.new",
    "expires_in": 3599,
    "scope": SCOPE,
    "token_type": "Bearer",
}
INVALID_GRANT = {
    "error": "invalid_grant",
    "error_description": "Token has been expired or revoked.",
}
REDIRECT_URI = "http://localhost:8000/api/v1/integrations/google_drive/callback"


def settings(**overrides):
    values = {
        "client_id": "gd_client",
        "client_secret": "shhh",
        "redirect_uri": REDIRECT_URI,
        "frontend_base_url": "http://localhost:3000",
        "env_prefix": "GOOGLE_DRIVE_",
    }
    values.update(overrides)
    return FakeSettings(**values)


def exchange_http(userinfo=WORKSPACE_USERINFO, token_response=TOKEN_RESPONSE, status=200):
    return FakeHttpClient(
        {
            ("POST", client.TOKEN_URL): (status, token_response),
            ("GET", client.USERINFO_URL): (200, userinfo),
        }
    )


def connector(http=None, **overrides) -> GoogleConnector:
    return GoogleConnector(
        google_descriptor("google_drive", "Google Drive", SCOPE),
        settings=settings(**overrides),
        http=http,
    )


def credential(**overrides) -> FakeCredential:
    values = {
        "provider": "google_drive",
        "provider_account_id": SUBJECT,
        "account_label": "goran@topoteretes.com",
        "provider_metadata": {"email": "goran@topoteretes.com"},
    }
    values.update(overrides)
    return FakeCredential(**values)


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


# ---- The descriptor factory and the base connector ---------------------------------


def test_google_descriptor_writes_the_shared_oauth_shape_down_once():
    scope = "openid email https://www.googleapis.com/auth/gmail.readonly"
    descriptor = google_descriptor("gmail", "Gmail", scope)

    assert (descriptor.provider, descriptor.display_name, descriptor.auth) == (
        "gmail",
        "Gmail",
        "oauth2",
    )
    assert descriptor.authorize_url == AUTHORIZE_URL
    assert descriptor.token_url == client.TOKEN_URL == "https://oauth2.googleapis.com/token"
    assert descriptor.scopes == scope
    assert descriptor.account_id_path == "userinfo.sub"
    assert descriptor.account_label_path == "userinfo.email"
    assert descriptor.signature_scheme is None
    assert descriptor.poll_interval_seconds == POLL_INTERVAL_SECONDS


def test_the_base_connector_is_the_identity_only_google_account():
    assert (CONNECTOR.provider, CONNECTOR.display_name, CONNECTOR.auth) == (
        "google",
        "Google",
        "oauth2",
    )
    assert DESCRIPTOR.scopes == IDENTITY_SCOPES == "openid email"
    assert isinstance(CONNECTOR, GoogleConnector)
    assert registry.tier_of(CONNECTOR) == 2
    assert CONNECTOR.webhook_verifier() is None
    assert type(CONNECTOR).supports_resource_selection() is False
    assert CONNECTOR.poll_targets(credential()) == []
    assert CONNECTOR.content_type is None
    # Built without settings: it is a base to derive from, not a configured provider.
    assert CONNECTOR.settings is None
    with pytest.raises(RuntimeError, match="without settings"):
        CONNECTOR.authorize_url("state")


def test_registering_the_base_lists_it_in_the_catalog():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("google") is CONNECTOR
    (entry,) = registry.catalog()
    assert entry == {
        "provider": "google",
        "display_name": "Google",
        "auth": "oauth2",
        "tier": 2,
        "has_webhooks": False,
        "supports_resource_selection": False,
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "content_type": None,
        "degraded": False,
    }


# ---- Consent ---------------------------------------------------------------------


def test_authorize_url_asks_for_offline_access_with_a_forced_consent_and_the_scope_only():
    url = urlsplit(connector().authorize_url("signed-state"))
    query = {key: value[0] for key, value in parse_qs(url.query).items()}

    assert f"{url.scheme}://{url.netloc}{url.path}" == AUTHORIZE_URL
    # Without both access_type and prompt Google reuses a prior consent and returns no
    # refresh token; include_granted_scopes would fold this grant together with a later
    # Gmail or Calendar one, and Google's revoke kills a whole grant.
    assert query == {
        "client_id": "gd_client",
        "redirect_uri": REDIRECT_URI,
        "state": "signed-state",
        "scope": SCOPE,
        "response_type": "code",
        "access_type": "offline",
        "prompt": "consent",
    }


def test_a_missing_setting_is_named_by_its_environment_variable():
    with pytest.raises(RuntimeError, match="GOOGLE_DRIVE_CLIENT_ID is not configured"):
        connector(client_id="").authorize_url("state")


# ---- Code exchange and installation -----------------------------------------------


@pytest.mark.asyncio
async def test_the_exchange_spends_the_fresh_token_on_one_userinfo_call():
    http = exchange_http()

    account = await connector(http=http).exchange_callback("auth-code", {"state": "ignored"})

    exchange, lookup = http.calls
    assert exchange.url == client.TOKEN_URL
    assert exchange.data == {
        "client_id": "gd_client",
        "client_secret": "shhh",
        "code": "auth-code",
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }
    # The fresh token is what proves the identity, so it is what gets spent.
    assert (lookup.method, lookup.url) == ("GET", client.USERINFO_URL)
    assert lookup.headers == {"Authorization": "Bearer ya29.access"}
    assert account == {**TOKEN_RESPONSE, "userinfo": WORKSPACE_USERINFO}


@pytest.mark.asyncio
async def test_a_rejected_exchange_is_a_runtime_error_that_names_the_reason():
    http = exchange_http(token_response=INVALID_GRANT, status=400)

    with pytest.raises(RuntimeError, match="invalid_grant"):
        await connector(http=http).exchange_callback("bad-code", {})

    assert [call.method for call in http.calls] == ["POST"]  # no lookup without a token


@pytest.mark.asyncio
async def test_a_failed_userinfo_lookup_names_the_status_only():
    http = FakeHttpClient(
        {
            ("POST", client.TOKEN_URL): (200, TOKEN_RESPONSE),
            ("GET", client.USERINFO_URL): (401, {"error": {"message": "secret body"}}),
        }
    )

    with pytest.raises(RuntimeError, match="userinfo lookup failed: HTTP 401") as caught:
        await connector(http=http).exchange_callback("auth-code", {})

    assert "secret" not in str(caught.value)


@pytest.mark.asyncio
async def test_an_exchange_without_an_access_token_is_refused_before_the_lookup():
    http = exchange_http(token_response={"token_type": "Bearer", "expires_in": 3599})

    with pytest.raises(RuntimeError, match="no access token"):
        await connector(http=http).exchange_callback("auth-code", {})

    assert [call.method for call in http.calls] == ["POST"]


def test_a_workspace_install_is_keyed_on_the_subject():
    installation = connector().parse_installation(
        {**TOKEN_RESPONSE, "userinfo": WORKSPACE_USERINFO}
    )

    assert installation.provider_account_id == SUBJECT
    assert installation.token_payload == {
        "access_token": "ya29.access",
        "refresh_token": "1//refresh",
    }
    assert installation.provider_metadata == {
        "email": "goran@topoteretes.com",
        "hosted_domain": "topoteretes.com",
        "account_type": "workspace",
        "scope": SCOPE,
    }
    assert installation.account_label == "goran@topoteretes.com"
    assert installation.scopes == SCOPE
    assert installation.auth_type == "oauth2"
    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(seconds=3500)


def test_an_account_without_a_hosted_domain_is_personal():
    installation = connector().parse_installation({**TOKEN_RESPONSE, "userinfo": PERSONAL_USERINFO})

    assert installation.provider_account_id == PERSONAL_USERINFO["sub"]
    assert installation.provider_metadata["account_type"] == "personal"
    assert installation.provider_metadata["hosted_domain"] is None


def test_secret_material_stays_out_of_the_cleartext_half():
    installation = connector().parse_installation(
        {**TOKEN_RESPONSE, "userinfo": WORKSPACE_USERINFO}
    )

    cleartext = f"{installation.provider_metadata} {installation.account_label}"
    assert "ya29" not in cleartext and "1//refresh" not in cleartext


def test_a_missing_refresh_token_is_logged_and_left_out_of_the_payload(caplog):
    response = {key: value for key, value in TOKEN_RESPONSE.items() if key != "refresh_token"}

    with caplog.at_level(logging.WARNING, logger="cognee_connector_google.connector"):
        installation = connector().parse_installation({**response, "userinfo": WORKSPACE_USERINFO})

    assert installation.token_payload == {"access_token": "ya29.access"}
    assert "without a refresh token" in caplog.text
    assert SUBJECT in caplog.text


def test_a_response_without_a_subject_is_refused():
    with pytest.raises(RuntimeError, match="userinfo.sub"):
        connector().parse_installation({**TOKEN_RESPONSE, "userinfo": {}})


def test_the_dataset_is_keyed_on_the_subject_not_the_email():
    before = connector().dataset_name(credential(account_label="goran@topoteretes.com"))
    after = connector().dataset_name(credential(account_label="renamed@topoteretes.com"))

    assert before == after == f"google_drive_{SUBJECT}"


# ---- Refresh and revoke ------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_refresh_keeps_the_stored_refresh_token_google_does_not_rotate():
    http = FakeHttpClient({("POST", client.TOKEN_URL): (200, REFRESH_RESPONSE)})

    refreshed = await connector(http=http).refresh(
        credential(), {"access_token": "ya29.old", "refresh_token": "1//refresh"}
    )

    assert http.last_call.data == {
        "client_id": "gd_client",
        "client_secret": "shhh",
        "grant_type": "refresh_token",
        "refresh_token": "1//refresh",
    }
    assert refreshed.token_payload == {"access_token": "ya29.new", "refresh_token": "1//refresh"}
    assert refreshed.scopes == SCOPE
    assert refreshed.token_expires_at > datetime.now(timezone.utc) + timedelta(seconds=3500)


@pytest.mark.asyncio
async def test_an_invalid_grant_on_refresh_is_a_runtime_error_for_the_host_to_revoke_on():
    http = FakeHttpClient({("POST", client.TOKEN_URL): (400, INVALID_GRANT)})

    # No webhook tells us the account pulled access; invalid_grant is the only signal. The
    # core vets a refresh answer like a code exchange, so the message is the exchange one.
    with pytest.raises(RuntimeError, match="token exchange failed: invalid_grant"):
        await connector(http=http).refresh(credential(), {"refresh_token": "1//refresh"})


@pytest.mark.asyncio
async def test_a_payload_without_a_refresh_token_has_nothing_to_refresh():
    http = FakeHttpClient()

    assert await connector(http=http).refresh(credential(), {"access_token": "a"}) is None
    assert http.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "revoked"),
    [
        ({"refresh_token": "r", "access_token": "a"}, "r"),
        ({"access_token": "a"}, "a"),
    ],
)
async def test_disconnect_revokes_the_refresh_token_in_preference_to_the_access_token(
    payload, revoked
):
    http = FakeHttpClient({("POST", client.REVOKE_URL): (200, {})})

    await connector(http=http).revoke_remote(credential(), payload)

    assert http.last_call.url == client.REVOKE_URL == "https://oauth2.googleapis.com/revoke"
    assert http.last_call.data == {"token": revoked}


@pytest.mark.asyncio
async def test_disconnect_revokes_nothing_remotely_when_there_is_no_token():
    http = FakeHttpClient()

    await connector(http=http).revoke_remote(credential(), {})

    assert http.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [RuntimeError("down"), (400, {"error": "invalid_token"})])
async def test_disconnect_is_never_blocked_by_google(answer, caplog):
    http = FakeHttpClient({("POST", client.REVOKE_URL): answer})

    with caplog.at_level(logging.WARNING, logger="cognee_connector_google.connector"):
        await connector(http=http).revoke_remote(credential(), {"refresh_token": "r"})

    assert "revoke" in caplog.text


# ---- The bearer token a sync runs with ------------------------------------------------


def test_a_payload_without_an_access_token_is_refused():
    with pytest.raises(RuntimeError, match=f"{SUBJECT} holds no access token"):
        connector()._bearer(credential(), {"refresh_token": "r"})


def test_the_bearer_token_is_the_stored_access_token():
    assert connector()._bearer(credential(), {"access_token": "ya29.ok"}) == "ya29.ok"
