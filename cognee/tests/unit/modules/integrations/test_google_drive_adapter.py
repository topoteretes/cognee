"""Unit tests for cognee.modules.integrations.google_drive.adapter.

Network calls are mocked at the module's own ``client`` seam rather than at
aiohttp, since that seam is the whole point of having a client module. What
is under test is the install and rotation policy: the two authorize-URL
parameters that decide whether this connector still works an hour later, the
personal-versus-Workspace split, and what a refresh writes back.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee.modules.integrations.google_drive import adapter as adapter_module
from cognee.modules.integrations.google_drive.adapter import (
    GoogleDriveIntegration,
    access_token_for,
)
from cognee.modules.integrations.google_drive.client import GoogleAuthError

_TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": "openid email https://www.googleapis.com/auth/drive.readonly",
}

_WORKSPACE_USERINFO = {
    "sub": "110000000000000000001",
    "email": "goran@topoteretes.com",
    "hd": "topoteretes.com",
}

_PERSONAL_USERINFO = {"sub": "110000000000000000002", "email": "someone@gmail.com"}


@pytest.fixture(autouse=True)
def _settings(monkeypatch):
    settings = (
        "cognee.modules.integrations.google_drive.google_drive_settings.google_drive_settings"
    )
    monkeypatch.setattr(f"{settings}.client_id", "gd_client")
    monkeypatch.setattr(f"{settings}.client_secret", "shhh")
    monkeypatch.setattr(
        f"{settings}.redirect_uri",
        "http://localhost:8000/api/v1/integrations/google_drive/callback",
    )
    monkeypatch.setattr(f"{settings}.state_secret", "state-secret")
    monkeypatch.setattr(f"{settings}.frontend_base_url", "http://localhost:3000")


def make_credential(**overrides):
    credential = SimpleNamespace(
        provider_account_id=_WORKSPACE_USERINFO["sub"],
        user_id="user-1",
        account_label="goran@topoteretes.com",
        auth_type="oauth2",
        scopes="openid email drive.readonly",
        token_expires_at=None,
    )
    for key, value in overrides.items():
        setattr(credential, key, value)
    return credential


def test_authorize_url_asks_for_a_refresh_token_and_a_separate_grant():
    url = GoogleDriveIntegration().authorize_url("signed-state")
    query = parse_qs(urlsplit(url).query)

    # Without both of these Google reuses a prior consent and returns no
    # refresh token, which strands the connection an hour after install.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    # Incremental authorization would fold this grant together with a later
    # Gmail or Calendar one, and Google's revoke kills a whole grant.
    assert "include_granted_scopes" not in query
    assert query["state"] == ["signed-state"]
    assert "drive.readonly" in query["scope"][0]


@pytest.mark.asyncio
async def test_exchange_callback_attaches_the_identity_of_who_authorized():
    integration = GoogleDriveIntegration()
    with (
        patch.object(
            adapter_module.client, "exchange_code", AsyncMock(return_value=_TOKEN_RESPONSE)
        ),
        patch.object(
            adapter_module.client,
            "fetch_userinfo",
            AsyncMock(return_value=_WORKSPACE_USERINFO),
        ) as fetch_userinfo,
    ):
        response = await integration.exchange_callback("auth-code", {})

    # The fresh token is what proves the identity, so it is what gets spent.
    fetch_userinfo.assert_awaited_once_with(_TOKEN_RESPONSE["access_token"])
    assert response["userinfo"] == _WORKSPACE_USERINFO


def test_parse_installation_keys_on_the_subject_and_marks_a_workspace_account():
    installation = GoogleDriveIntegration().parse_installation(
        {**_TOKEN_RESPONSE, "userinfo": _WORKSPACE_USERINFO}
    )

    assert installation.provider_account_id == _WORKSPACE_USERINFO["sub"]
    assert installation.account_label == "goran@topoteretes.com"
    assert installation.provider_metadata["account_type"] == "workspace"
    assert installation.provider_metadata["hosted_domain"] == "topoteretes.com"
    # Secret material stays out of the cleartext half.
    assert set(installation.token_payload) == {"access_token", "refresh_token"}
    assert "access_token" not in installation.provider_metadata


def test_parse_installation_marks_an_account_without_a_hosted_domain_as_personal():
    installation = GoogleDriveIntegration().parse_installation(
        {**_TOKEN_RESPONSE, "userinfo": _PERSONAL_USERINFO}
    )

    assert installation.provider_metadata["account_type"] == "personal"
    assert installation.provider_metadata["hosted_domain"] is None


def test_parse_installation_refuses_a_response_without_a_subject():
    with pytest.raises(ValueError):
        GoogleDriveIntegration().parse_installation({**_TOKEN_RESPONSE, "userinfo": {}})


def test_parse_installation_survives_a_missing_refresh_token():
    # Google withholds it when a prior consent is reused. The install still
    # works; it just cannot outlive the access token.
    response = {k: v for k, v in _TOKEN_RESPONSE.items() if k != "refresh_token"}
    installation = GoogleDriveIntegration().parse_installation(
        {**response, "userinfo": _PERSONAL_USERINFO}
    )

    assert set(installation.token_payload) == {"access_token"}


@pytest.mark.asyncio
async def test_refresh_carries_forward_the_fields_upsert_would_otherwise_clear():
    credential = make_credential()
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"access_token": "old", "refresh_token": "1//refresh"},
        ),
        patch.object(
            adapter_module.client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "ya29.new", "expires_in": 3599}),
        ),
        patch.object(adapter_module, "upsert_credential", AsyncMock()) as upsert,
    ):
        await GoogleDriveIntegration().refresh(credential)

    written = upsert.await_args.kwargs
    # account_label, auth_type and scopes are still assigned unconditionally
    # by upsert_credential, so omitting them would wipe them on every hourly
    # rotation. provider_metadata and workspace_id survive omission, which is
    # why they are deliberately absent here.
    assert written["account_label"] == credential.account_label
    assert written["auth_type"] == credential.auth_type
    assert written["scopes"] == credential.scopes
    assert "provider_metadata" not in written
    assert "workspace_id" not in written
    # Google issues no new refresh token, so the stored one is carried over.
    assert written["token_payload"] == {
        "access_token": "ya29.new",
        "refresh_token": "1//refresh",
    }


@pytest.mark.asyncio
async def test_refresh_revokes_the_credential_when_the_account_pulled_access():
    credential = make_credential()
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"refresh_token": "1//refresh"},
        ),
        patch.object(
            adapter_module.client,
            "refresh_access_token",
            AsyncMock(side_effect=GoogleAuthError("token refresh", "invalid_grant")),
        ),
        patch.object(adapter_module, "revoke_credential_by_account", AsyncMock()) as revoke,
        pytest.raises(GoogleAuthError),
    ):
        await GoogleDriveIntegration().refresh(credential)

    # There is no webhook telling us the account revoked us, so an
    # invalid_grant is the only signal, and it must not leave the connection
    # showing as healthy.
    revoke.assert_awaited_once_with("google_drive", credential.provider_account_id)


@pytest.mark.asyncio
async def test_refresh_does_not_revoke_on_a_transient_failure():
    credential = make_credential()
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"refresh_token": "1//refresh"},
        ),
        patch.object(
            adapter_module.client,
            "refresh_access_token",
            AsyncMock(side_effect=GoogleAuthError("token refresh", "http_503")),
        ),
        patch.object(adapter_module, "revoke_credential_by_account", AsyncMock()) as revoke,
        pytest.raises(GoogleAuthError),
    ):
        await GoogleDriveIntegration().refresh(credential)

    revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_without_a_stored_refresh_token_fails_loudly():
    with (
        patch.object(adapter_module, "decrypt_token_payload", return_value={"access_token": "a"}),
        pytest.raises(RuntimeError, match="reconnect"),
    ):
        await GoogleDriveIntegration().refresh(make_credential())


@pytest.mark.asyncio
async def test_revoke_remote_never_raises():
    credential = make_credential()
    # A disconnect must complete even when the remote revoke cannot.
    with patch.object(adapter_module, "decrypt_token_payload", side_effect=RuntimeError("boom")):
        await GoogleDriveIntegration().revoke_remote(credential)


@pytest.mark.asyncio
async def test_access_token_for_rotates_an_expiring_token_and_rereads_the_row():
    stale = make_credential(token_expires_at=datetime.now(timezone.utc) + timedelta(seconds=30))
    fresh = make_credential(token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    with (
        patch.object(GoogleDriveIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(
            adapter_module, "get_credential_by_account", AsyncMock(return_value=fresh)
        ) as reread,
        patch.object(
            adapter_module, "decrypt_token_payload", return_value={"access_token": "ya29.new"}
        ) as decrypt,
    ):
        token = await access_token_for(stale)

    refresh.assert_awaited_once()
    # The row has to be re-read: refresh writes through its own session, so
    # the instance we were handed still holds the pre-rotation ciphertext.
    reread.assert_awaited_once()
    decrypt.assert_called_once_with(fresh)
    assert token == "ya29.new"


@pytest.mark.asyncio
async def test_access_token_for_leaves_a_healthy_token_alone():
    credential = make_credential(token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    with (
        patch.object(GoogleDriveIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(
            adapter_module, "decrypt_token_payload", return_value={"access_token": "ya29.ok"}
        ),
    ):
        token = await access_token_for(credential)

    refresh.assert_not_awaited()
    assert token == "ya29.ok"


@pytest.mark.asyncio
async def test_access_token_for_treats_a_naive_expiry_as_utc():
    # SQLite stores no timezone, so the column reads back naive. Comparing a
    # naive value against an aware now() raises, which would turn every sync
    # on the default backend into a TypeError.
    naive_expiry = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(tzinfo=None)
    credential = make_credential(token_expires_at=naive_expiry)

    with (
        patch.object(GoogleDriveIntegration, "refresh", AsyncMock()) as refresh,
        patch.object(
            adapter_module, "decrypt_token_payload", return_value={"access_token": "ya29.ok"}
        ),
    ):
        token = await access_token_for(credential)

    refresh.assert_not_awaited()
    assert token == "ya29.ok"
