"""Contract tests for the Dropbox connector: ``CONNECTOR`` against Dropbox's real shapes.

The token response and account lookup are the shapes cognee's adapter tests carry
(cognee.tests.unit.modules.integrations.test_dropbox_integration). Dropbox's calls go through
:mod:`cognee_connector_dropbox.client`, which these tests patch; a ``FakeHttpClient`` is wired
in anyway, to prove the core's form-encoded client never sees a call.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest
from fake_dropbox import ACCOUNT

from cognee_connector_dropbox import CONNECTOR, client
from cognee_connector_dropbox.connector import (
    DESCRIPTOR,
    POLL_INTERVAL_SECONDS,
    SCOPES,
    DropboxConnector,
)
from cognee_connector_dropbox.selection import (
    NestedSelectionError,
    SelectedFolderNotFoundError,
    refuse_nested_selection,
    resource_name_for,
)
from cognee_connectors_core.testing import FakeCredential, FakeHttpClient, FakeSettings

EMAIL = "person@example.com"
# cognee's name for ACCOUNT's dataset with EMAIL stored: a literal, because the name is a
# stored identifier that must survive every refactor in both copies.
DATASET = "dropbox_person_example_com_43138a9075"


def token_response(**overrides):
    return {
        "access_token": "sl.access",
        "refresh_token": "refresh",
        "expires_in": 14400,
        "account_id": ACCOUNT,
        "scope": "files.metadata.read",
        "token_type": "bearer",
        "uid": "12345",
        **overrides,
    }


def settings(**overrides):
    values = {
        "client_id": "app-key",
        "client_secret": "app-secret",
        "redirect_uri": "https://cognee.test/api/v1/integrations/dropbox/callback",
        "frontend_base_url": "https://cognee.test",
        "env_prefix": "DROPBOX_",
        "max_file_size_mb": 50.0,
    }
    values.update(overrides)
    return FakeSettings(**values)


def connector(http=None, **overrides) -> DropboxConnector:
    return DropboxConnector(DESCRIPTOR, settings=settings(**overrides), http=http)


def credential(**overrides) -> FakeCredential:
    values = {
        "provider": "dropbox",
        "provider_account_id": ACCOUNT,
        "account_label": EMAIL,
        "provider_metadata": {"email": EMAIL, "selected_folder_ids": []},
    }
    values.update(overrides)
    return FakeCredential(**values)


TOKENS = {"access_token": "sl.a", "refresh_token": "r"}


def test_the_descriptor_names_dropbox_and_polls():
    assert CONNECTOR.descriptor is DESCRIPTOR
    assert (DESCRIPTOR.provider, DESCRIPTOR.display_name, DESCRIPTOR.auth) == (
        "dropbox",
        "Dropbox",
        "oauth2",
    )
    assert DESCRIPTOR.token_url == "https://api.dropboxapi.com/oauth2/token"
    assert DESCRIPTOR.signature_scheme is None
    assert DESCRIPTOR.poll_interval_seconds == POLL_INTERVAL_SECONDS
    assert CONNECTOR.supports_resource_selection()


def test_authorize_url_asks_for_offline_access_and_the_read_scopes():
    url = connector().authorize_url("signed-state")

    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.netloc == "www.dropbox.com" and parts.path == "/oauth2/authorize"
    assert query["token_access_type"] == ["offline"] and query["response_type"] == ["code"]
    assert (
        query["scope"] == [SCOPES] == ["files.metadata.read files.content.read account_info.read"]
    )
    assert query["state"] == ["signed-state"] and query["client_id"] == ["app-key"]
    assert query["redirect_uri"] == ["https://cognee.test/api/v1/integrations/dropbox/callback"]
    assert "app-secret" not in url


@pytest.mark.asyncio
async def test_connect_keys_the_account_by_id_and_starts_with_no_folder():
    http = FakeHttpClient()
    with (
        patch.object(client, "exchange_code", AsyncMock(return_value=token_response())) as code,
        patch.object(
            client,
            "get_current_account",
            AsyncMock(return_value={"account_id": ACCOUNT, "email": EMAIL}),
        ) as lookup,
    ):
        response = await connector(http).exchange_callback("code-1", {})

    code.assert_awaited_once_with(
        "code-1",
        client_id="app-key",
        client_secret="app-secret",
        redirect_uri="https://cognee.test/api/v1/integrations/dropbox/callback",
    )
    lookup.assert_awaited_once_with("sl.access")
    assert http.calls == []
    installation = connector().parse_installation(response)
    assert installation.provider_account_id == ACCOUNT
    assert installation.token_payload == {"access_token": "sl.access", "refresh_token": "refresh"}
    assert installation.provider_metadata == {"email": EMAIL, "selected_folder_ids": []}
    assert installation.account_label == EMAIL
    assert installation.scopes == "files.metadata.read"
    assert installation.token_expires_at is not None
    assert "sl.access" not in str(installation.provider_metadata)


@pytest.mark.asyncio
async def test_a_token_for_another_account_than_the_lookup_is_refused():
    with (
        patch.object(client, "exchange_code", AsyncMock(return_value=token_response())),
        patch.object(
            client, "get_current_account", AsyncMock(return_value={"account_id": "dbid:B"})
        ),
        pytest.raises(ValueError, match="different accounts"),
    ):
        await connector().exchange_callback("code", {})


@pytest.mark.parametrize("account_id", [None, "", "12345", "AAH4f99", 7])
def test_a_token_response_without_a_dbid_account_is_refused(account_id):
    with pytest.raises(ValueError, match="account_id"):
        connector().parse_installation(token_response(account_id=account_id))


@pytest.mark.parametrize("expires_in", [None, 0, -1, "14400", True])
def test_only_a_positive_integer_lifetime_sets_an_expiry(expires_in):
    installation = connector().parse_installation(token_response(expires_in=expires_in))
    assert installation.token_expires_at is None


@pytest.mark.asyncio
async def test_a_refresh_keeps_the_refresh_token_dropbox_does_not_rotate():
    with patch.object(
        client,
        "refresh_access_token",
        AsyncMock(return_value={"access_token": "b", "expires_in": 14400}),
    ) as refresh:
        refreshed = await connector().refresh(credential(scopes="s"), TOKENS)

    refresh.assert_awaited_once_with("r", client_id="app-key", client_secret="app-secret")
    assert refreshed.token_payload == {"access_token": "b", "refresh_token": "r"}
    assert refreshed.scopes == "s"
    assert refreshed.token_expires_at > datetime.now(timezone.utc) + timedelta(hours=3)


@pytest.mark.asyncio
async def test_a_refresh_answer_without_a_lifetime_is_refused():
    with (
        patch.object(client, "refresh_access_token", AsyncMock(return_value={"access_token": "b"})),
        pytest.raises(client.DropboxIntegrationError, match="no_expires_in"),
    ):
        await connector().refresh(credential(), TOKENS)


@pytest.mark.asyncio
async def test_a_refresh_dropbox_refuses_is_raised_for_the_host_to_revoke():
    refused = client.DropboxIntegrationError("token refresh", "invalid_grant")
    with (
        patch.object(client, "refresh_access_token", AsyncMock(side_effect=refused)),
        pytest.raises(client.DropboxIntegrationError) as error,
    ):
        await connector().refresh(credential(), TOKENS)
    assert error.value.code == "invalid_grant"


@pytest.mark.asyncio
async def test_nothing_to_refresh_without_a_refresh_token():
    with patch.object(client, "refresh_access_token", AsyncMock()) as refresh:
        assert await connector().refresh(credential(), {"access_token": "a"}) is None
    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_revokes_the_token_and_survives_a_failed_revoke():
    with patch.object(client, "revoke_token", AsyncMock()) as revoke:
        await connector().revoke_remote(credential(), {"access_token": "sl.a"})
    revoke.assert_awaited_once_with("sl.a")

    failed = client.DropboxIntegrationError("token revoke", "http_401")
    with patch.object(client, "revoke_token", AsyncMock(side_effect=failed)):
        await connector().revoke_remote(credential(), {"access_token": "sl.a"})


@pytest.mark.asyncio
async def test_disconnect_revokes_with_a_fresh_token_when_the_stored_one_expired():
    expired = credential(token_expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
    with (
        patch.object(
            client,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "sl.fresh", "expires_in": 14400}),
        ) as refresh,
        patch.object(client, "revoke_token", AsyncMock()) as revoke,
    ):
        await connector().revoke_remote(
            expired, {"access_token": "sl.expired", "refresh_token": "r"}
        )

    refresh.assert_awaited_once_with("r", client_id="app-key", client_secret="app-secret")
    revoke.assert_awaited_once_with("sl.fresh")


@pytest.mark.asyncio
async def test_a_token_far_from_its_expiry_revokes_itself():
    fresh = credential(token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1))
    with (
        patch.object(client, "refresh_access_token", AsyncMock()) as refresh,
        patch.object(client, "revoke_token", AsyncMock()) as revoke,
    ):
        await connector().revoke_remote(fresh, TOKENS)

    refresh.assert_not_awaited()
    revoke.assert_awaited_once_with("sl.a")


def test_the_dataset_is_named_as_cognee_names_it():
    assert connector().dataset_name(credential()) == DATASET
    assert connector().dataset_name(credential(provider_metadata={})) == (
        "dropbox_account_43138a9075"
    )


def test_the_dataset_name_matches_cognees_own_formula():
    ingestion = pytest.importorskip("cognee.modules.integrations.ingestion")
    for email in (EMAIL, "", "A.B@x.io", "  "):
        expected = ingestion.dataset_name_for_account("dropbox", email, ACCOUNT)
        actual = connector().dataset_name(credential(provider_metadata={"email": email}))
        assert actual == expected, email


@pytest.mark.asyncio
async def test_list_resources_offers_top_level_folders_by_id_with_their_path():
    folders = [
        {".tag": "folder", "id": "id:w", "path_lower": "/work", "path_display": "/Work"},
        {".tag": "folder", "id": "id:n", "path_lower": "/notes", "path_display": "/Notes"},
        {".tag": "folder", "path_lower": "/noid"},
    ]
    with patch.object(client, "list_folders", AsyncMock(return_value=folders)) as listing:
        resources = await connector().list_resources(credential(), TOKENS)

    listing.assert_awaited_once_with("sl.a")
    assert [(r.id, r.name, r.attributes) for r in resources] == [
        ("id:n", "/Notes", {"path": "/Notes"}),
        ("id:w", "/Work", {"path": "/Work"}),
    ]


def _folders(*metas):
    return patch.object(client, "get_folders", AsyncMock(return_value=list(metas)))


@pytest.mark.asyncio
async def test_sync_scope_reads_the_stored_selection_and_hands_out_the_token():
    stored = credential(provider_metadata={"email": EMAIL, "selected_folder_ids": ["id:w", "id:n"]})
    with _folders({"path_lower": "/notes"}, {"path_lower": "/work"}) as lookup:
        scope = await connector(max_file_size_mb=5.0).sync_scope(stored, TOKENS)

    lookup.assert_awaited_once_with("sl.a", ["id:n", "id:w"])
    assert scope.items == ["id:n", "id:w"]
    assert scope.dataset_name == DATASET
    assert scope.credential == "sl.a" and scope.credential_refreshable is True
    assert scope.options == {"account_id": ACCOUNT, "max_file_size_mb": 5.0}


@pytest.mark.asyncio
async def test_an_empty_selection_syncs_nothing_and_makes_no_request():
    with _folders() as lookup:
        scope = await connector().sync_scope(credential(), TOKENS)
    lookup.assert_not_awaited()
    assert scope.items == []


@pytest.mark.asyncio
async def test_an_explicit_empty_resources_list_is_refused():
    with pytest.raises(ValueError, match="empty"):
        await connector().sync_scope(credential(), TOKENS, resources=[])


@pytest.mark.parametrize("selection", [None, "id:n", ["/Notes"], [""], ["id:"], ["id:a b"]])
@pytest.mark.asyncio
async def test_a_selection_that_is_not_folder_ids_is_refused(selection):
    stored = credential(provider_metadata={"selected_folder_ids": selection})
    if selection is None:
        # No stored selection is an empty one, as in cognee's sync.
        with _folders():
            assert (await connector().sync_scope(stored, TOKENS)).items == []
        return
    with _folders() as lookup, pytest.raises(ValueError, match="folder ids"):
        await connector().sync_scope(stored, TOKENS)
    lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_folder_inside_another_selected_folder_is_refused():
    with (
        _folders({"path_lower": "/notes"}, {"path_lower": "/notes/sub"}),
        pytest.raises(NestedSelectionError),
    ):
        await connector().sync_scope(credential(), TOKENS, resources=["id:a", "id:b"])


@pytest.mark.asyncio
async def test_a_sibling_whose_name_starts_like_another_folder_is_not_nested():
    with _folders({"path_lower": "/notes"}, {"path_lower": "/notes-old"}):
        scope = await connector().sync_scope(credential(), TOKENS, resources=["id:a", "id:b"])
    assert scope.items == ["id:a", "id:b"]


@pytest.mark.asyncio
async def test_a_missing_folder_passes_a_sync_and_is_refused_when_a_selection_is_saved():
    with _folders({"path_lower": "/notes"}, None):
        scope = await connector().sync_scope(credential(), TOKENS, resources=["id:a", "id:gone"])
    assert scope.items == ["id:a", "id:gone"]

    with (
        _folders({"path_lower": "/notes"}, None),
        pytest.raises(SelectedFolderNotFoundError),
    ):
        await refuse_nested_selection("sl.a", ["id:a", "id:gone"], refuse_missing=True)


@pytest.mark.asyncio
async def test_a_payload_without_an_access_token_is_refused():
    with pytest.raises(RuntimeError, match="no access token"):
        await connector().sync_scope(credential(), {}, resources=["id:a"])


@pytest.mark.asyncio
async def test_a_single_folder_is_not_looked_up_for_nesting_on_a_sync():
    """One folder cannot sit inside another selected one, so a sync (every poll) asks Dropbox
    nothing for the nesting check."""
    with _folders({"path_lower": "/notes"}) as lookup:
        scope = await connector().sync_scope(credential(), TOKENS, resources=["id:a"])
    lookup.assert_not_awaited()
    assert scope.items == ["id:a"]


@pytest.mark.asyncio
async def test_a_single_folder_is_still_resolved_when_a_selection_is_saved():
    with _folders(None) as lookup, pytest.raises(SelectedFolderNotFoundError):
        await refuse_nested_selection("sl.a", ["id:gone"], refuse_missing=True)
    lookup.assert_awaited_once_with("sl.a", ["id:gone"])


USER_ID = "22222222-2222-2222-2222-222222222222"


def test_the_resource_name_is_cognees_table_and_cursor_name():
    """A literal, like the dataset name: the table and cursor a host takes over from cognee."""
    assert resource_name_for(USER_ID, ACCOUNT, "id:notes") == "dropbox_42d93adafa7b96c4c99470ce"
    assert resource_name_for(UUID(USER_ID), ACCOUNT, "id:notes") == (
        "dropbox_42d93adafa7b96c4c99470ce"
    )
    # Non-ASCII ids are escaped in the JSON the digest is taken over, as in cognee.
    assert (
        resource_name_for(USER_ID, ACCOUNT, "id:\u00e9_\u2603")
        == "dropbox_df6ea6171015843cc180b808"
    )


def test_the_resource_name_matches_cognees_own_formula():
    pytest.importorskip(
        "cognee.tasks.ingestion.connectors.dropbox",
        reason="the installed cognee has no in-tree Dropbox connector to compare with",
    )
    ingestion = pytest.importorskip("cognee.modules.integrations.ingestion")
    for user_id in (USER_ID, UUID(USER_ID)):
        for folder in ("id:notes", "id:a4ayc_80_OEAAAAAAAAAXw", "", "id:\u00e9", "id:\u2603"):
            owner = SimpleNamespace(user_id=user_id, provider_account_id=ACCOUNT)
            expected = ingestion.resource_name("dropbox", owner, folder)
            assert resource_name_for(user_id, ACCOUNT, folder) == expected, (user_id, folder)


@pytest.mark.asyncio
async def test_a_refused_token_reaches_the_source_with_one_folder_and_raises_with_two():
    """One folder asks Dropbox nothing in sync_scope, so a token Dropbox refuses is the
    source's to report; two or more folders meet it in the nesting lookup."""
    refused = client.DropboxIntegrationError("folder lookup", "http_401")
    with patch.object(client, "get_folders", AsyncMock(side_effect=refused)) as lookup:
        scope = await connector().sync_scope(credential(), TOKENS, resources=["id:a"])
        lookup.assert_not_awaited()
        assert scope.items == ["id:a"]

        with pytest.raises(client.DropboxIntegrationError):
            await connector().sync_scope(credential(), TOKENS, resources=["id:a", "id:b"])
