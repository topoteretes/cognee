"""Contract tests for the Google Drive connector: ``CONNECTOR`` against Drive's real shapes.

The token-response and userinfo fixtures are the ones ``test_google_drive_adapter.py``
carries; Google's token and userinfo endpoints are faked with ``FakeHttpClient``, and the
Drive REST calls at the client module's own seams, as the adapter tests fake them. Nothing
here imports cognee.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from cognee_connector_google import GoogleConnector
from cognee_connector_google import client as google_client
from cognee_connector_google_drive import CONNECTOR, client
from cognee_connector_google_drive.connector import (
    DESCRIPTOR,
    ROOT_RESOURCE_ID,
    SCOPE,
    GoogleDriveConnector,
)
from cognee_connectors_core import registry
from cognee_connectors_core.base import ConnectorResource
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings

SUBJECT = "110000000000000000001"
TOKEN_RESPONSE = {
    "access_token": "ya29.access",
    "refresh_token": "1//refresh",
    "expires_in": 3599,
    "scope": SCOPE,
    "token_type": "Bearer",
}
WORKSPACE_USERINFO = {"sub": SUBJECT, "email": "goran@topoteretes.com", "hd": "topoteretes.com"}
PERSONAL_USERINFO = {"sub": "110000000000000000002", "email": "someone@gmail.com"}
REDIRECT_URI = "http://localhost:8000/api/v1/integrations/google_drive/callback"
FOLDER_MIME = "application/vnd.google-apps.folder"


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


def connector(http=None, **overrides) -> GoogleDriveConnector:
    return GoogleDriveConnector(DESCRIPTOR, settings=settings(**overrides), http=http)


def credential(**overrides) -> FakeCredential:
    values = {
        "provider": "google_drive",
        "provider_account_id": SUBJECT,
        "account_label": "goran@topoteretes.com",
        "provider_metadata": {"email": "goran@topoteretes.com", "selected_folder_ids": []},
    }
    values.update(overrides)
    return FakeCredential(**values)


def drives(*pages):
    """``client.list_drives`` answering one page per call, the last one without a cursor."""
    answers = [
        {"drives": [{"id": drive_id, "name": name} for drive_id, name in page]} for page in pages
    ]
    for answer in answers[:-1]:
        answer["nextPageToken"] = "next"
    return AsyncMock(side_effect=answers or [{"drives": []}])


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.reset_registry()
    yield
    registry.reset_registry()


# ---- Identity and catalog ------------------------------------------------------------


def test_the_descriptor_names_drive_as_a_polled_oauth2_provider_with_a_read_only_scope():
    assert (DESCRIPTOR.provider, DESCRIPTOR.display_name, DESCRIPTOR.auth) == (
        "google_drive",
        "Google Drive",
        "oauth2",
    )
    assert DESCRIPTOR.authorize_url == "https://accounts.google.com/o/oauth2/v2/auth"
    assert DESCRIPTOR.token_url == google_client.TOKEN_URL
    # Anything wider than drive.readonly would let a leaked token change files.
    assert DESCRIPTOR.scopes == "openid email https://www.googleapis.com/auth/drive.readonly"
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds == 3600


def test_the_connector_is_tier_two_with_a_picker_and_no_webhooks():
    assert (CONNECTOR.provider, CONNECTOR.display_name, CONNECTOR.auth) == (
        "google_drive",
        "Google Drive",
        "oauth2",
    )
    assert isinstance(CONNECTOR, GoogleDriveConnector)
    assert isinstance(CONNECTOR, GoogleConnector)
    assert registry.tier_of(CONNECTOR) == 2
    assert CONNECTOR.webhook_verifier() is None
    assert type(CONNECTOR).supports_resource_selection() is True
    assert CONNECTOR.poll_targets(credential()) == []
    assert CONNECTOR.content_type is None


def test_the_connector_reads_the_real_settings_object():
    from cognee_connector_google_drive.settings import google_drive_settings

    assert CONNECTOR.settings is google_drive_settings


def test_registering_the_connector_lists_it_in_the_catalog():
    registry.use_connector(CONNECTOR)

    assert registry.get_connector("google_drive") is CONNECTOR
    (entry,) = registry.catalog()
    assert entry == {
        "provider": "google_drive",
        "display_name": "Google Drive",
        "auth": "oauth2",
        "tier": 2,
        "has_webhooks": False,
        "supports_resource_selection": True,
        "poll_interval_seconds": 3600,
        "content_type": None,
        "degraded": False,
    }


# ---- Connect ---------------------------------------------------------------------------


def test_authorize_url_asks_for_a_refresh_token_and_a_separate_grant():
    query = parse_qs(urlsplit(connector().authorize_url("signed-state")).query)

    # Without both of these Google reuses a prior consent and returns no refresh token,
    # which strands the connection an hour after install.
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    # Incremental authorization would fold this grant together with a later Gmail or
    # Calendar one, and Google's revoke kills a whole grant.
    assert "include_granted_scopes" not in query
    assert query["state"] == ["signed-state"]
    assert query["client_id"] == ["gd_client"]
    assert query["redirect_uri"] == [REDIRECT_URI]
    assert query["scope"] == [SCOPE]


def test_a_missing_setting_is_named_by_its_environment_variable():
    with pytest.raises(RuntimeError, match="GOOGLE_DRIVE_CLIENT_ID is not configured"):
        connector(client_id="").authorize_url("state")


@pytest.mark.asyncio
async def test_exchange_callback_attaches_the_identity_of_who_authorized():
    http = FakeHttpClient(
        {
            ("POST", google_client.TOKEN_URL): (200, TOKEN_RESPONSE),
            ("GET", google_client.USERINFO_URL): (200, WORKSPACE_USERINFO),
        }
    )

    account = await connector(http=http).exchange_callback("auth-code", {})
    installation = connector().parse_installation(account)

    exchange, lookup = http.calls
    assert exchange.data["redirect_uri"] == REDIRECT_URI
    assert lookup.headers == {"Authorization": "Bearer ya29.access"}
    assert installation.provider_account_id == SUBJECT
    assert installation.account_label == "goran@topoteretes.com"
    assert installation.provider_metadata == {
        "email": "goran@topoteretes.com",
        "hosted_domain": "topoteretes.com",
        "account_type": "workspace",
        "scope": SCOPE,
    }
    assert set(installation.token_payload) == {"access_token", "refresh_token"}
    assert "access_token" not in installation.provider_metadata
    assert installation.scopes == SCOPE
    assert installation.token_expires_at > datetime.now(timezone.utc) + timedelta(seconds=3500)


def test_parse_installation_marks_an_account_without_a_hosted_domain_as_personal():
    installation = CONNECTOR.parse_installation({**TOKEN_RESPONSE, "userinfo": PERSONAL_USERINFO})

    assert installation.provider_metadata["account_type"] == "personal"
    assert installation.provider_metadata["hosted_domain"] is None


def test_parse_installation_refuses_a_response_without_a_subject():
    with pytest.raises(RuntimeError, match="userinfo.sub"):
        CONNECTOR.parse_installation({**TOKEN_RESPONSE, "userinfo": {}})


def test_the_dataset_is_keyed_on_the_subject_and_survives_a_renamed_account():
    before = CONNECTOR.dataset_name(credential(account_label="goran@topoteretes.com"))
    after = CONNECTOR.dataset_name(credential(account_label="renamed@topoteretes.com"))

    assert before == after == f"google_drive_{SUBJECT}"


# ---- Resources ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_picker_offers_my_drive_the_shared_drives_and_every_folder():
    folders = AsyncMock(
        side_effect=[
            {
                "files": [{"id": "folder-a", "name": "Plans", "mimeType": FOLDER_MIME}],
                "nextPageToken": "p2",
            },
            {
                "files": [
                    {"id": "folder-b", "mimeType": FOLDER_MIME, "driveId": "shared-1"},
                    {"name": "no id"},
                ]
            },
        ]
    )
    with (
        patch.object(client, "list_drives", drives([("shared-1", "Team")], [("shared-2", "")])),
        patch.object(client, "list_folders", folders),
    ):
        resources = await connector().list_resources(credential(), {"access_token": "tok"})

    assert all(isinstance(resource, ConnectorResource) for resource in resources)
    assert [(item.id, item.name) for item in resources] == [
        ("root", "My Drive"),
        ("shared-1", "Team"),
        ("shared-2", "Shared drive"),
        ("folder-a", "Plans"),
        ("folder-b", "Untitled folder"),
    ]
    assert resources[0].description == "Files in the account root, including nested folders"
    assert resources[0].attributes == {"mime_type": FOLDER_MIME, "root": True}
    assert resources[1].attributes == {"mime_type": FOLDER_MIME, "shared_drive": True, "root": True}
    assert resources[4].description is None
    assert resources[4].attributes == {"mime_type": FOLDER_MIME, "drive_id": "shared-1"}
    assert folders.await_args_list[1].args == ("tok", "p2")


@pytest.mark.asyncio
async def test_the_picker_refuses_a_payload_without_an_access_token():
    with (
        patch.object(client, "list_drives", AsyncMock()) as list_drives,
        pytest.raises(RuntimeError, match=f"{SUBJECT} holds no access token"),
    ):
        await connector().list_resources(credential(), {"refresh_token": "r"})

    list_drives.assert_not_awaited()


# ---- Sync scope ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_scope_without_a_selection_is_my_drive_plus_every_shared_drive():
    expires = datetime.now(timezone.utc) + timedelta(hours=1)
    with patch.object(client, "list_drives", drives([("shared-1", "Team")], [("shared-2", "Ops")])):
        scope = await connector().sync_scope(
            credential(token_expires_at=expires), {"access_token": "tok", "refresh_token": "r"}
        )

    assert scope.items == [ROOT_RESOURCE_ID, "shared-1", "shared-2"]
    assert scope.item_resource_ids == scope.items
    assert scope.dataset_name == f"google_drive_{SUBJECT}"
    assert scope.content_type is None
    assert scope.credential == "tok"
    assert scope.credential_expires_at == expires
    assert scope.credential_refreshable is True
    assert scope.options == {
        "folder_ids": ["root", "shared-1", "shared-2"],
        "shared_drive_ids": ["shared-1", "shared-2"],
    }


@pytest.mark.asyncio
async def test_sync_scope_takes_the_folders_a_host_names_once_each_and_marks_shared_drives():
    with patch.object(client, "list_drives", drives([("shared-1", "Team")])):
        scope = await connector().sync_scope(
            credential(),
            {"access_token": "tok"},
            resources=["folder-a", "shared-1", "folder-a"],
        )

    assert scope.items == ["folder-a", "shared-1"]
    assert scope.item_resource_ids == ["folder-a", "shared-1"]
    assert scope.options == {
        "folder_ids": ["folder-a", "shared-1"],
        "shared_drive_ids": ["shared-1"],
    }
    assert scope.credential_refreshable is False


@pytest.mark.asyncio
async def test_sync_scope_refuses_an_empty_selection_before_any_request():
    with (
        patch.object(client, "list_drives", AsyncMock()) as list_drives,
        pytest.raises(ValueError, match="empty"),
    ):
        await connector().sync_scope(credential(), {"access_token": "tok"}, resources=[])

    list_drives.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_scope_refuses_a_payload_without_an_access_token():
    with (
        patch.object(client, "list_drives", AsyncMock()) as list_drives,
        pytest.raises(RuntimeError, match="holds no access token"),
    ):
        await connector().sync_scope(credential(), {})

    list_drives.assert_not_awaited()
