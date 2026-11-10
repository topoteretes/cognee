"""Gmail, Drive and Calendar share one Google OAuth contract, and each keeps its own scope.

Every provider asks for offline access with a forced consent, never folds in earlier
grants, refreshes five minutes before expiry, carries the refresh token forward, revokes
its credential when Google answers ``invalid_grant`` and keeps its dataset name stable.
The cases run against all three providers so one cannot drift from the others; a dataset
name is a stored identifier, so its golden values are fixed strings.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee.modules.integrations import ingestion
from cognee.modules.integrations.gmail import adapter as gmail_adapter
from cognee.modules.integrations.gmail import sync as gmail_sync
from cognee.modules.integrations.gmail.gmail_settings import google_gmail_settings
from cognee.modules.integrations.google import client as google_client
from cognee.modules.integrations.google import oauth_integration
from cognee.modules.integrations.google_calendar import adapter as calendar_adapter
from cognee.modules.integrations.google_calendar import sync as calendar_sync
from cognee.modules.integrations.google_calendar.google_calendar_settings import (
    google_calendar_settings,
)
from cognee.modules.integrations.google_drive import adapter as drive_adapter
from cognee.modules.integrations.google_drive import sync as drive_sync
from cognee.modules.integrations.google_drive.google_drive_settings import google_drive_settings
from cognee.modules.integrations.registry import get_integration

SUBJECT = "110000000000000000001"
EXPIRES_IN = 3599
REFRESH_MARGIN = timedelta(minutes=5)


def provider(
    name, display_name, adapter_module, integration_cls, scope, selection_key, env_prefix, settings
):
    return SimpleNamespace(
        name=name,
        display_name=display_name,
        adapter=adapter_module,
        integration_cls=integration_cls,
        scope=scope,
        selection_key=selection_key,
        env_prefix=env_prefix,
        settings=settings,
    )


PROVIDERS = [
    provider(
        "gmail",
        "Gmail",
        gmail_adapter,
        gmail_adapter.GoogleGmailIntegration,
        "openid email https://www.googleapis.com/auth/gmail.readonly",
        "selected_label_ids",
        "GOOGLE_GMAIL_",
        google_gmail_settings,
    ),
    provider(
        "google_drive",
        "Google Drive",
        drive_adapter,
        drive_adapter.GoogleDriveIntegration,
        "openid email https://www.googleapis.com/auth/drive.readonly",
        "selected_folder_ids",
        "GOOGLE_DRIVE_",
        google_drive_settings,
    ),
    provider(
        "google_calendar",
        "Google Calendar",
        calendar_adapter,
        calendar_adapter.GoogleCalendarIntegration,
        "openid email https://www.googleapis.com/auth/calendar.readonly",
        "selected_calendar_ids",
        "GOOGLE_CALENDAR_",
        google_calendar_settings,
    ),
]


@pytest.fixture(params=PROVIDERS, ids=lambda case: case.name)
def case(request, monkeypatch):
    settings = request.param.settings
    monkeypatch.setattr(settings, "client_id", "client-id")
    monkeypatch.setattr(settings, "client_secret", "client-secret")
    monkeypatch.setattr(settings, "redirect_uri", "http://localhost:8000/callback")
    monkeypatch.setattr(settings, "state_secret", "state-secret")
    monkeypatch.setattr(settings, "frontend_base_url", "http://localhost:3000")
    return request.param


def credential(**overrides):
    values = {
        "provider_account_id": SUBJECT,
        "scopes": "old scope",
        "token_expires_at": None,
        "provider_metadata": {"email": "person@example.com"},
    }
    return SimpleNamespace(**{**values, **overrides})


# --- consent and installation ---


def test_authorize_url_asks_for_offline_access_with_this_providers_scope_only(case):
    url = case.integration_cls().authorize_url("signed-state")

    parts = urlsplit(url)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == (
        "https://accounts.google.com/o/oauth2/v2/auth"
    )
    assert {key: value[0] for key, value in parse_qs(parts.query).items()} == {
        "client_id": "client-id",
        "redirect_uri": "http://localhost:8000/callback",
        "response_type": "code",
        "state": "signed-state",
        "scope": case.scope,
        "access_type": "offline",
        "prompt": "consent",
    }


@pytest.mark.asyncio
async def test_the_code_is_exchanged_with_this_providers_oauth_client(case):
    exchange = AsyncMock(return_value={"access_token": "a"})
    with patch.object(google_client, "exchange_code", exchange):
        assert await case.integration_cls().exchange_code("code") == {"access_token": "a"}

    exchange.assert_awaited_once_with(
        "code",
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="http://localhost:8000/callback",
    )


@pytest.mark.asyncio
async def test_exchange_callback_attaches_the_identity_of_who_authorized(case):
    userinfo = {"sub": SUBJECT, "email": "person@example.com"}
    with (
        patch.object(google_client, "exchange_code", AsyncMock(return_value={"access_token": "a"})),
        patch.object(google_client, "fetch_userinfo", AsyncMock(return_value=userinfo)) as fetch,
    ):
        response = await case.integration_cls().exchange_callback("code", {})

    fetch.assert_awaited_once_with("a")
    assert response == {"access_token": "a", "userinfo": userinfo}


def test_a_workspace_install_is_keyed_on_the_subject_and_starts_with_nothing_selected(case):
    token_response = {
        "access_token": "a",
        "refresh_token": "r",
        "expires_in": EXPIRES_IN,
        "scope": case.scope,
        "userinfo": {"sub": SUBJECT, "email": "person@example.com", "hd": "example.com"},
    }

    installation = case.integration_cls().parse_installation(token_response)

    assert installation.provider_account_id == SUBJECT
    assert installation.token_payload == {"access_token": "a", "refresh_token": "r"}
    assert installation.provider_metadata == {
        "email": "person@example.com",
        "hosted_domain": "example.com",
        "account_type": "workspace",
        "scope": case.scope,
        case.selection_key: [],
    }
    assert installation.account_label == "person@example.com"
    assert installation.scopes == case.scope
    assert installation.auth_type == "oauth2"
    assert installation.token_expires_at is not None
    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(seconds=3500)


@pytest.mark.parametrize("expires_in", [None, 0])
def test_an_install_without_an_expiry_has_no_expiry_date(case, expires_in):
    token_response = {
        "access_token": "a",
        "expires_in": expires_in,
        "userinfo": {"sub": SUBJECT},
    }

    installation = case.integration_cls().parse_installation(token_response)

    assert installation.token_expires_at is None
    assert installation.provider_metadata["account_type"] == "personal"


def test_a_missing_refresh_token_is_logged_and_left_out_of_the_payload(case, caplog):
    token_response = {"access_token": "a", "userinfo": {"sub": SUBJECT}}

    with caplog.at_level("WARNING", logger=oauth_integration.logger.name):
        installation = case.integration_cls().parse_installation(token_response)

    assert installation.token_payload == {"access_token": "a"}
    assert "without a refresh token" in caplog.text
    assert SUBJECT in caplog.text


def test_a_response_without_a_subject_is_refused(case):
    with pytest.raises(ValueError, match="no subject"):
        case.integration_cls().parse_installation({"access_token": "a", "userinfo": {}})


def test_the_state_secret_and_frontend_url_come_from_this_providers_settings(case, monkeypatch):
    integration = case.integration_cls()
    assert integration.state_signing_secret() == "state-secret"
    assert integration.frontend_base_url() == "http://localhost:3000"

    monkeypatch.setattr(case.settings, "state_secret", "")
    with pytest.raises(RuntimeError, match=f"{case.env_prefix}STATE_SECRET is not configured"):
        integration.state_signing_secret()


# --- refresh ---


@pytest.mark.asyncio
async def test_refresh_carries_the_refresh_token_forward_and_keeps_the_old_scopes(case):
    with (
        patch.object(
            oauth_integration,
            "decrypt_token_payload",
            return_value={"refresh_token": "r", "x": "y"},
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "new", "expires_in": EXPIRES_IN}),
        ) as refresh,
        patch.object(oauth_integration, "update_refreshed_credential", AsyncMock()) as write,
    ):
        stored = credential()
        await case.integration_cls().refresh(stored)

    refresh.assert_awaited_once_with("r", client_id="client-id", client_secret="client-secret")
    assert write.await_args.args == (stored,)
    written = write.await_args.kwargs
    assert set(written) == {"token_payload", "token_expires_at", "scopes"}
    assert written["token_payload"] == {"access_token": "new", "refresh_token": "r"}
    assert written["scopes"] == "old scope"
    assert written["token_expires_at"] > datetime.now(timezone.utc) + timedelta(seconds=3500)


@pytest.mark.asyncio
async def test_refresh_takes_the_scopes_google_reports(case):
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "r"}
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "new", "scope": "narrower"}),
        ),
        patch.object(oauth_integration, "update_refreshed_credential", AsyncMock()) as write,
    ):
        await case.integration_cls().refresh(credential())

    assert write.await_args.kwargs["scopes"] == "narrower"
    assert write.await_args.kwargs["token_expires_at"] is None


@pytest.mark.asyncio
async def test_refresh_revokes_the_credential_when_google_says_invalid_grant(case, caplog):
    stored = credential()
    with (
        caplog.at_level("WARNING", logger=oauth_integration.logger.name),
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "r"}
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(side_effect=google_client.GoogleAuthError("token refresh", "invalid_grant")),
        ),
        patch.object(oauth_integration, "revoke_credential_if_current", AsyncMock()) as revoke,
        patch.object(oauth_integration, "update_refreshed_credential", AsyncMock()) as write,
        pytest.raises(google_client.GoogleAuthError),
    ):
        await case.integration_cls().refresh(stored)

    revoke.assert_awaited_once_with(stored)
    write.assert_not_awaited()
    # Nothing else tells an operator why the connection stopped syncing.
    assert f"{case.display_name} access for account {SUBJECT} was revoked" in caplog.text


@pytest.mark.asyncio
async def test_refresh_leaves_the_credential_alone_on_a_transient_failure(case):
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"refresh_token": "r"}
        ),
        patch.object(
            google_client,
            "refresh_access_token",
            AsyncMock(side_effect=google_client.GoogleAuthError("token refresh", "http_503")),
        ),
        patch.object(oauth_integration, "revoke_credential_if_current", AsyncMock()) as revoke,
        pytest.raises(google_client.GoogleAuthError),
    ):
        await case.integration_cls().refresh(credential())

    revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_without_a_stored_refresh_token_asks_to_reconnect(case):
    with (
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "a"}
        ),
        patch.object(google_client, "refresh_access_token", AsyncMock()) as refresh,
        pytest.raises(
            RuntimeError,
            match=f"^{case.display_name} credential for account {SUBJECT} holds no refresh token",
        ),
    ):
        await case.integration_cls().refresh(credential())

    refresh.assert_not_awaited()


# --- access token ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("time_left", "refreshes"),
    [
        (REFRESH_MARGIN - timedelta(seconds=30), True),
        (REFRESH_MARGIN + timedelta(seconds=30), False),
        (-timedelta(hours=1), True),
    ],
)
async def test_the_token_is_rotated_five_minutes_before_it_expires(case, time_left, refreshes):
    stored = credential(token_expires_at=datetime.now(timezone.utc) + time_left)
    with (
        patch.object(case.integration_cls, "refresh", AsyncMock()) as refresh,
        patch.object(
            oauth_integration, "require_active_credential", AsyncMock(side_effect=lambda c: c)
        ),
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "tok"}
        ),
    ):
        assert await case.adapter.access_token_for(stored) == "tok"

    assert refresh.await_count == (1 if refreshes else 0)


@pytest.mark.asyncio
async def test_a_rotated_token_is_read_from_the_row_written_by_the_refresh(case):
    stale = credential(token_expires_at=datetime.now(timezone.utc))
    fresh = credential(token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
    with (
        patch.object(case.integration_cls, "refresh", AsyncMock()),
        patch.object(
            oauth_integration, "require_active_credential", AsyncMock(side_effect=[stale, fresh])
        ),
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "new"}
        ) as decrypt,
    ):
        assert await case.adapter.access_token_for(stale) == "new"

    decrypt.assert_called_once_with(fresh)


@pytest.mark.asyncio
async def test_a_naive_expiry_is_read_as_utc(case):
    # SQLite returns the column without a timezone.
    expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)
    with (
        patch.object(case.integration_cls, "refresh", AsyncMock()) as refresh,
        patch.object(
            oauth_integration, "require_active_credential", AsyncMock(side_effect=lambda c: c)
        ),
        patch.object(
            oauth_integration, "decrypt_token_payload", return_value={"access_token": "tok"}
        ),
    ):
        await case.adapter.access_token_for(credential(token_expires_at=expiry))

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_credential_without_an_access_token_fails_loudly(case):
    with (
        patch.object(
            oauth_integration, "require_active_credential", AsyncMock(side_effect=lambda c: c)
        ),
        patch.object(oauth_integration, "decrypt_token_payload", return_value={}),
        pytest.raises(RuntimeError, match=f"{SUBJECT} holds no access token"),
    ):
        await case.adapter.access_token_for(credential())


# --- disconnect ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "revoked"),
    [
        ({"refresh_token": "r", "access_token": "a"}, "r"),
        ({"access_token": "a"}, "a"),
    ],
)
async def test_disconnect_revokes_the_refresh_token_in_preference_to_the_access_token(
    case, payload, revoked
):
    with (
        patch.object(oauth_integration, "decrypt_token_payload", return_value=payload),
        patch.object(google_client, "revoke_token", AsyncMock()) as revoke,
    ):
        await case.integration_cls().revoke_remote(credential())

    revoke.assert_awaited_once_with(revoked)


@pytest.mark.asyncio
async def test_disconnect_revokes_nothing_remotely_when_there_is_no_token(case):
    with (
        patch.object(oauth_integration, "decrypt_token_payload", return_value={}),
        patch.object(google_client, "revoke_token", AsyncMock()) as revoke,
    ):
        await case.integration_cls().revoke_remote(credential())

    revoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["decrypt", "revoke"])
async def test_disconnect_is_never_blocked_by_google(case, failing, caplog):
    decrypt = {"side_effect": RuntimeError("boom")} if failing == "decrypt" else {}
    revoke = {"side_effect": RuntimeError("down")} if failing == "revoke" else {}
    with (
        patch.object(
            oauth_integration,
            "decrypt_token_payload",
            return_value={"refresh_token": "r"},
            **decrypt,
        ),
        patch.object(google_client, "revoke_token", AsyncMock(**revoke)),
        caplog.at_level("ERROR", logger=oauth_integration.logger.name),
    ):
        await case.integration_cls().revoke_remote(credential())

    assert SUBJECT in caplog.text


# --- identity ---


@pytest.mark.parametrize(
    ("name", "integration_cls", "selection_key"),
    [
        ("gmail", gmail_adapter.GoogleGmailIntegration, "selected_label_ids"),
        ("google_drive", drive_adapter.GoogleDriveIntegration, "selected_folder_ids"),
        ("google_calendar", calendar_adapter.GoogleCalendarIntegration, "selected_calendar_ids"),
    ],
)
def test_each_provider_is_registered_under_its_own_name_and_selection_key(
    name, integration_cls, selection_key
):
    integration = get_integration(name)

    assert type(integration) is integration_cls
    assert integration.provider == name
    assert integration.resource_selection_key == selection_key


# Only Calendar refuses a null selection: the connector reads it as the primary calendar.
def test_only_calendar_refuses_a_null_selection():
    assert gmail_adapter.GoogleGmailIntegration.resource_selection_allows_all is True
    assert drive_adapter.GoogleDriveIntegration.resource_selection_allows_all is True
    assert calendar_adapter.GoogleCalendarIntegration.resource_selection_allows_all is False


# --- dataset names ---
#
# A dataset name is a stored identifier: changing the slug rule or the length of the hash
# suffix would orphan every dataset already synced. The values below are the names
# already stored for synced accounts.

DATASET_NAMES = [
    ("Person@Example.com", "", "person_example_com"),
    ("a_.b@example.com", "", "a__b_example_com"),
    ("", "", "account"),
    ("person@example.com", "subject", "person_example_com_a9491f4c1b"),
    ("John.Doe+x@Gmail.com", SUBJECT, "john_doe_x_gmail_com_39c326d15a"),
    ("", SUBJECT, "account_39c326d15a"),
    ("---@@@", "x", "account_2d711642b7"),
]


@pytest.mark.parametrize("prefix", ["gmail", "google_drive", "google_calendar"])
@pytest.mark.parametrize(("email", "account_id", "suffix"), DATASET_NAMES)
def test_dataset_names_are_byte_identical_to_the_ones_already_stored(
    prefix, email, account_id, suffix
):
    assert ingestion.dataset_name_for_account(prefix, email, account_id) == f"{prefix}_{suffix}"


def test_the_adapter_and_the_sync_name_the_dataset_the_same_way(case):
    stored = credential(provider_metadata={"email": "Person@Example.com"})
    sync_module = {
        "gmail": gmail_sync,
        "google_drive": drive_sync,
        "google_calendar": calendar_sync,
    }[case.name]

    name = case.integration_cls().dataset_name(stored)

    assert name == sync_module._dataset_name(stored)
    assert name == f"{case.name}_person_example_com_{name.rsplit('_', 1)[1]}"
    assert len(name.rsplit("_", 1)[1]) == 10


def test_a_credential_without_an_email_still_gets_a_dataset_name(case):
    name = case.integration_cls().dataset_name(credential(provider_metadata=None))

    assert name.startswith(f"{case.name}_account_")


# --- provider definition ---


@pytest.mark.parametrize("missing", ["display_name", "scope", "settings", "dataset_prefix"])
def test_a_provider_missing_a_class_attribute_fails_when_it_is_defined(missing):
    # Without this the gap shows up as an AttributeError in the middle of a consent flow.
    attributes = {
        "display_name": "Example",
        "scope": "openid email",
        "settings": google_gmail_settings,
        "dataset_prefix": "example",
    }
    del attributes[missing]

    with pytest.raises(TypeError, match=f"^Incomplete must set {missing}$"):
        type("Incomplete", (oauth_integration.GoogleOAuthIntegration,), attributes)
