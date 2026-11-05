"""Google Calendar's OAuth adapter asks for read-only access and keeps connect opt-in.

Network calls are faked at the adapter's own ``client`` seams. What is held here is
the policy: the scope and authorize parameters, the empty default selection, the
calendar picker and when a refresh revokes the credential.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee.modules.integrations.google import client as google_client
from cognee.modules.integrations.google import oauth_integration
from cognee.modules.integrations.google_calendar import client as calendar_client
from cognee.modules.integrations.google_calendar.adapter import (
    GoogleCalendarIntegration,
    access_token_for,
)
from cognee.modules.integrations.google_calendar.google_calendar_settings import (
    GoogleCalendarOAuthSettings,
)
from cognee.modules.integrations.registry import get_integration

SETTINGS = (
    "cognee.modules.integrations.google_calendar.google_calendar_settings.google_calendar_settings"
)
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": "openid email https://www.googleapis.com/auth/calendar.readonly",
}
USERINFO = {
    "sub": "110000000000000000001",
    "email": "goran@topoteretes.com",
    "hd": "topoteretes.com",
}


@pytest.fixture(autouse=True)
def active_credential():
    with patch.object(
        oauth_integration, "require_active_credential", AsyncMock(side_effect=lambda c: c)
    ):
        yield


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    monkeypatch.setattr(f"{SETTINGS}.client_id", "cal_client")
    monkeypatch.setattr(f"{SETTINGS}.client_secret", "shhh")
    monkeypatch.setattr(
        f"{SETTINGS}.redirect_uri",
        "http://localhost:8000/api/v1/integrations/google_calendar/callback",
    )
    monkeypatch.setattr(f"{SETTINGS}.state_secret", "state-secret")
    monkeypatch.setattr(f"{SETTINGS}.frontend_base_url", "http://localhost:3000")


def test_package_import_registers_the_provider():
    assert isinstance(get_integration("google_calendar"), GoogleCalendarIntegration)


def test_authorize_url_asks_for_read_only_calendar_access_and_a_refresh_token():
    query = parse_qs(urlsplit(GoogleCalendarIntegration().authorize_url("signed-state")).query)

    # Anything wider than calendar.readonly would let a leaked token edit meetings.
    assert query["scope"] == ["openid email https://www.googleapis.com/auth/calendar.readonly"]
    # Without both of these Google reuses a prior consent and returns no refresh token.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert "include_granted_scopes" not in query
    assert query["state"] == ["signed-state"]
    assert query["client_id"] == ["cal_client"]


def test_settings_read_the_google_calendar_prefix(monkeypatch):
    # The prefix is shared with the SDK connector's settings, so it must not drift.
    monkeypatch.setenv("GOOGLE_CALENDAR_CLIENT_ID", "from-env")
    assert GoogleCalendarOAuthSettings().client_id == "from-env"


def test_state_secret_and_frontend_url_come_from_their_own_settings():
    integration = GoogleCalendarIntegration()
    assert integration.state_signing_secret() == "state-secret"
    assert integration.frontend_base_url() == "http://localhost:3000"


@pytest.mark.asyncio
async def test_exchange_code_sends_the_registered_redirect_uri():
    with patch.object(
        google_client, "exchange_code", AsyncMock(return_value=TOKEN_RESPONSE)
    ) as exchange:
        await GoogleCalendarIntegration().exchange_code("auth-code")

    assert exchange.await_args.kwargs["redirect_uri"] == (
        "http://localhost:8000/api/v1/integrations/google_calendar/callback"
    )


@pytest.mark.asyncio
async def test_exchange_callback_attaches_the_identity_of_who_authorized():
    with (
        patch.object(google_client, "exchange_code", AsyncMock(return_value=TOKEN_RESPONSE)),
        patch.object(
            google_client, "fetch_userinfo", AsyncMock(return_value=USERINFO)
        ) as fetch_userinfo,
    ):
        response = await GoogleCalendarIntegration().exchange_callback("auth-code", {})

    # The fresh token is what proves the identity, so it is what gets spent.
    fetch_userinfo.assert_awaited_once_with(TOKEN_RESPONSE["access_token"])
    assert response["userinfo"] == USERINFO


def test_unconfigured_settings_name_the_environment_variable(monkeypatch):
    monkeypatch.setattr(f"{SETTINGS}.client_id", "")
    with pytest.raises(RuntimeError, match="GOOGLE_CALENDAR_CLIENT_ID"):
        GoogleCalendarIntegration().authorize_url("state")


def test_parse_installation_starts_with_no_calendar_selected():
    installation = GoogleCalendarIntegration().parse_installation(
        {**TOKEN_RESPONSE, "userinfo": USERINFO}
    )

    assert installation.provider_account_id == USERINFO["sub"]
    assert installation.provider_metadata["selected_calendar_ids"] == []
    assert installation.provider_metadata["account_type"] == "workspace"
    assert installation.provider_metadata["hosted_domain"] == "topoteretes.com"
    assert set(installation.token_payload) == {"access_token", "refresh_token"}
    assert "access_token" not in installation.provider_metadata


def test_parse_installation_keeps_the_grant_the_account_label_and_the_expiry():
    installation = GoogleCalendarIntegration().parse_installation(
        {**TOKEN_RESPONSE, "userinfo": USERINFO}
    )

    assert installation.provider_metadata["scope"] == TOKEN_RESPONSE["scope"]
    assert installation.scopes == TOKEN_RESPONSE["scope"]
    assert installation.account_label == USERINFO["email"]
    assert installation.token_expires_at > datetime.now(timezone.utc)


def test_the_router_selection_key_is_seeded_by_the_installation():
    installation = GoogleCalendarIntegration().parse_installation(
        {**TOKEN_RESPONSE, "userinfo": USERINFO}
    )

    # The router writes the picker's choice under this key and sync reads it back.
    assert GoogleCalendarIntegration.resource_selection_key in installation.provider_metadata


def test_parse_installation_refuses_a_response_without_a_subject():
    with pytest.raises(ValueError, match="no subject"):
        GoogleCalendarIntegration().parse_installation({**TOKEN_RESPONSE, "userinfo": {}})


def test_missing_refresh_token_is_logged_and_the_install_still_works(caplog):
    response = {k: v for k, v in TOKEN_RESPONSE.items() if k != "refresh_token"}
    with caplog.at_level("WARNING", logger=oauth_integration.logger.name):
        installation = GoogleCalendarIntegration().parse_installation(
            {**response, "userinfo": USERINFO}
        )

    assert set(installation.token_payload) == {"access_token"}
    assert "without a refresh token" in caplog.text


@pytest.mark.asyncio
async def test_calendar_picker_uses_the_client_with_the_current_access_token():
    credential = SimpleNamespace(token_expires_at=None)
    calendars = [
        {
            "id": "me@example.com",
            "summary": "me@example.com",
            "summaryOverride": "Work",
            "primary": True,
            "accessRole": "owner",
        },
        {"id": "team@group.calendar.google.com", "summary": "Team", "description": "Standups"},
        {"summary": "missing id"},
    ]
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "t"}
        ),
        patch.object(calendar_client, "list_calendars", AsyncMock(return_value=calendars)) as lst,
    ):
        resources = await GoogleCalendarIntegration().list_resources(credential)

    lst.assert_awaited_once_with("t")
    assert [(r["id"], r["name"]) for r in resources] == [
        ("me@example.com", "Work"),
        ("team@group.calendar.google.com", "Team"),
    ]
    assert resources[0]["attributes"] == {"primary": True, "access_role": "owner"}
    assert resources[1]["description"] == "Standups"
    assert resources[1]["attributes"]["primary"] is False


# The refresh margin is 5 minutes: a token already expired or expiring inside it is
# refreshed before listing, one with more time left is used as it is.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expires_in", [timedelta(minutes=-1), timedelta(minutes=2), timedelta(minutes=10)]
)
async def test_token_inside_the_refresh_margin_is_refreshed_before_listing_calendars(expires_in):
    stale = expires_in < timedelta(minutes=5)
    credential = SimpleNamespace(
        token_expires_at=datetime.now(timezone.utc) + expires_in,
        provider_account_id="subject",
    )
    refreshed = SimpleNamespace(token_expires_at=None)
    with (
        patch.object(GoogleCalendarIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(
            oauth_integration,
            "require_active_credential",
            AsyncMock(side_effect=[credential, refreshed] if stale else [credential]),
        ),
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "fresh"}
        ) as decrypt,
        patch.object(calendar_client, "list_calendars", AsyncMock(return_value=[])) as lst,
    ):
        assert await GoogleCalendarIntegration().list_resources(credential) == []

    if stale:
        refresh.assert_awaited_once_with(credential)
        decrypt.assert_called_once_with(refreshed)
    else:
        refresh.assert_not_awaited()
        decrypt.assert_called_once_with(credential)
    lst.assert_awaited_once_with("fresh")


@pytest.mark.asyncio
async def test_access_token_for_treats_a_naive_expiry_as_utc():
    # SQLite stores no timezone, so the column reads back naive. Comparing it with an
    # aware now() would raise, and every sync on the default backend would fail.
    naive_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)
    credential = SimpleNamespace(token_expires_at=naive_expiry, provider_account_id="subject")
    with (
        patch.object(GoogleCalendarIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "t"}
        ),
    ):
        assert await access_token_for(credential) == "t"

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_credential_without_an_access_token_fails_loudly():
    credential = SimpleNamespace(token_expires_at=None, provider_account_id="subject")
    with (
        patch.object(oauth_integration, "decrypt_token_payload", return_value={}),
        pytest.raises(RuntimeError, match="no access token"),
    ):
        await access_token_for(credential)


@pytest.mark.asyncio
async def test_refresh_keeps_the_stored_refresh_token():
    credential = SimpleNamespace(provider_account_id="subject", scopes="calendar")
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "1//refresh"}
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "ya29.new", "expires_in": 3599}),
        ),
        patch.object(oauth_integration, "update_refreshed_credential", AsyncMock()) as update,
    ):
        await GoogleCalendarIntegration().refresh(credential)

    # Google issues no new refresh token on a refresh, so the old one is carried over.
    assert update.await_args.kwargs["token_payload"] == {
        "access_token": "ya29.new",
        "refresh_token": "1//refresh",
    }
    assert update.await_args.kwargs["scopes"] == "calendar"


@pytest.mark.asyncio
async def test_refresh_stores_the_scope_google_returns():
    credential = SimpleNamespace(provider_account_id="subject", scopes="calendar")
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "r"}
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "a", "expires_in": 10, "scope": "narrower"}),
        ),
        patch.object(oauth_integration, "update_refreshed_credential", AsyncMock()) as update,
    ):
        await GoogleCalendarIntegration().refresh(credential)

    assert update.await_args.kwargs["scopes"] == "narrower"


@pytest.mark.asyncio
async def test_refresh_without_a_refresh_token_asks_to_reconnect():
    credential = SimpleNamespace(provider_account_id="subject")
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "a"}
        ),
        pytest.raises(RuntimeError, match="reconnect"),
    ):
        await GoogleCalendarIntegration().refresh(credential)


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "revoked"), [("invalid_grant", True), ("http_503", False)])
async def test_refresh_revokes_only_when_google_says_the_grant_is_gone(code, revoked):
    credential = SimpleNamespace(provider_account_id="subject")
    error = google_client.GoogleAuthError("token refresh", code)
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "1//refresh"}
        ),
        patch.object(google_client, "refresh_access_token", AsyncMock(side_effect=error)),
        patch.object(oauth_integration, "revoke_credential_if_current", AsyncMock()) as revoke,
        pytest.raises(google_client.GoogleAuthError),
    ):
        await GoogleCalendarIntegration().refresh(credential)

    # No webhook tells us the account pulled access, so invalid_grant is the only
    # signal; a transient failure must not disconnect a healthy account.
    assert revoke.await_count == (1 if revoked else 0)


@pytest.mark.asyncio
async def test_revoke_remote_never_raises():
    credential = SimpleNamespace(provider_account_id="subject")
    # A disconnect must complete even when the remote revoke cannot.
    with patch.object(oauth_integration, "decrypt_token_payload", side_effect=RuntimeError("boom")):
        await GoogleCalendarIntegration().revoke_remote(credential)


@pytest.mark.asyncio
async def test_revoke_remote_prefers_the_refresh_token():
    credential = SimpleNamespace(provider_account_id="subject")
    with (
        patch.object(
            oauth_integration,
            "decrypt_token_payload",
            return_value={"access_token": "a", "refresh_token": "r"},
        ),
        patch.object(google_client, "revoke_token", AsyncMock()) as revoke,
    ):
        await GoogleCalendarIntegration().revoke_remote(credential)

    revoke.assert_awaited_once_with("r")
