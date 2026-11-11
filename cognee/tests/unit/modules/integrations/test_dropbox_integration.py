"""The Dropbox integration module: connect, picker, selection, Sync now and disconnect.

Provider calls are mocked at ``dropbox.client``; ingestion is mocked at ``ingestion``. Pause,
resume and schedule are the shared router's and the sync policies' (SDK-1434), tested there;
here only that a scheduled run reaches this adapter's ``sync_now``.
"""

from contextlib import ExitStack
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

import pytest

from cognee.modules.integrations.dropbox import adapter as adapter_module
from cognee.modules.integrations.dropbox import client as client_module
from cognee.modules.integrations.dropbox import sync as sync_module
from cognee.modules.integrations.dropbox.adapter import DropboxIntegration
from cognee.modules.integrations.dropbox.dropbox_settings import dropbox_settings

ACCOUNT = "dbid:AAA"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for name, value in {
        "client_id": "app-key",
        "client_secret": "app-secret",
        "redirect_uri": "https://cognee.test/api/v1/integrations/dropbox/callback",
        "state_secret": "state",
        "frontend_base_url": "https://cognee.test",
    }.items():
        monkeypatch.setattr(dropbox_settings, name, value)


def credential(**metadata):
    return SimpleNamespace(
        id=UUID("11111111-1111-1111-1111-111111111111"),
        provider="dropbox",
        provider_account_id=ACCOUNT,
        user_id=UUID("22222222-2222-2222-2222-222222222222"),
        provider_metadata={"email": "person@example.com", **metadata},
        token_expires_at=None,
        scopes=None,
    )


def test_the_adapter_is_registered_and_starts_unselected():
    import cognee.modules.integrations.dropbox
    from cognee.modules.integrations.registry import get_integration

    integration = get_integration("dropbox")
    assert isinstance(integration, DropboxIntegration)
    assert integration.resource_selection_key == "selected_folder_ids"
    assert integration.resource_selection_allows_all is False


def test_authorize_url_asks_for_offline_access_and_the_read_scopes():
    url = DropboxIntegration().authorize_url("signed-state")
    parts = urlsplit(url)
    query = parse_qs(parts.query)
    assert parts.netloc == "www.dropbox.com" and parts.path == "/oauth2/authorize"
    assert query["token_access_type"] == ["offline"]
    assert query["scope"] == ["files.metadata.read files.content.read account_info.read"]
    assert query["state"] == ["signed-state"] and query["client_id"] == ["app-key"]
    assert "app-secret" not in url


@pytest.mark.asyncio
async def test_connect_keys_the_account_by_id_and_starts_with_no_folder():
    token_response = {
        "access_token": "sl.access",
        "refresh_token": "refresh",
        "expires_in": 14400,
        "account_id": ACCOUNT,
        "scope": "files.metadata.read",
    }
    with (
        patch.object(client_module, "exchange_code", AsyncMock(return_value=token_response)),
        patch.object(
            client_module,
            "get_current_account",
            AsyncMock(return_value={"account_id": ACCOUNT, "email": "person@example.com"}),
        ),
    ):
        response = await DropboxIntegration().exchange_callback("code", {})

    installation = DropboxIntegration().parse_installation(response)
    assert installation.provider_account_id == ACCOUNT
    assert installation.token_payload == {"access_token": "sl.access", "refresh_token": "refresh"}
    assert installation.provider_metadata == {
        "email": "person@example.com",
        "selected_folder_ids": [],
    }
    assert installation.token_expires_at is not None
    assert "sl.access" not in str(installation.provider_metadata)


@pytest.mark.asyncio
async def test_a_token_for_another_account_than_the_lookup_is_refused():
    with (
        patch.object(
            client_module,
            "exchange_code",
            AsyncMock(return_value={"access_token": "t", "account_id": ACCOUNT}),
        ),
        patch.object(
            client_module, "get_current_account", AsyncMock(return_value={"account_id": "dbid:B"})
        ),
        pytest.raises(ValueError, match="different accounts"),
    ):
        await DropboxIntegration().exchange_callback("code", {})


@pytest.mark.asyncio
async def test_list_resources_offers_folders_by_id_with_their_path():
    folders = [
        {".tag": "folder", "id": "id:w", "path_lower": "/work", "path_display": "/Work"},
        {".tag": "folder", "id": "id:n", "path_lower": "/notes", "path_display": "/Notes"},
    ]
    with (
        patch.object(adapter_module, "access_token_for", AsyncMock(return_value="t")),
        patch.object(client_module, "list_folders", AsyncMock(return_value=folders)),
    ):
        resources = await DropboxIntegration().list_resources(credential())

    assert [(r["id"], r["name"]) for r in resources] == [("id:n", "/Notes"), ("id:w", "/Work")]


@pytest.fixture
def sync_mocks():
    with ExitStack() as stack:
        mocks = SimpleNamespace(
            source=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
            remember=AsyncMock(return_value=SimpleNamespace(status="completed")),
            retire=AsyncMock(),
            record=AsyncMock(),
            folder=AsyncMock(
                side_effect=lambda _t, folder_id: {
                    ".tag": "folder",
                    "path_lower": {"id:n": "/notes", "id:w": "/work", "id:sub": "/notes/sub"}[
                        folder_id
                    ],
                }
            ),
            service=Mock(),
            token=AsyncMock(return_value="token"),
        )
        for patcher in (
            patch.object(
                sync_module.ingestion,
                "require_active_credential",
                AsyncMock(side_effect=lambda credential: credential),
            ),
            patch.object(sync_module.ingestion, "retire_resources", mocks.retire),
            patch.object(sync_module.ingestion, "source_factory", return_value=mocks.source),
            patch.object(sync_module.ingestion, "build_service", return_value=mocks.service),
            patch.object(sync_module.client, "get_folder", mocks.folder),
            patch.object(adapter_module, "access_token_for", mocks.token),
            patch.object(
                import_module("cognee.api.v1.remember.remember"), "remember", mocks.remember
            ),
            patch("cognee.modules.integrations.credentials.record_sync_result", mocks.record),
            patch(
                "cognee.modules.integrations.sync_progress.track_sync",
                _fake_track_sync,
            ),
            patch("cognee.modules.integrations.sync_progress.report_sync_progress", AsyncMock()),
            patch("cognee.modules.integrations.sync_progress.get_current_sync_progress", Mock()),
            patch("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner")),
        ):
            stack.enter_context(patcher)
        yield mocks


class _fake_track_sync:
    def __init__(self, *_args):
        pass

    async def __aenter__(self):
        return SimpleNamespace(report=AsyncMock(), finish=AsyncMock())

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_sync_now_runs_one_source_per_folder_then_retires_the_rest(sync_mocks):
    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:w", "id:n", "id:w"]))

    calls = [c.kwargs for c in sync_mocks.source.call_args_list]
    assert [c["folder_id"] for c in calls] == ["id:n", "id:w"]
    assert all(c["account_id"] == ACCOUNT and "service" not in c for c in calls)
    assert all(c["resource_name"].startswith("dropbox_") for c in calls)
    retained = sync_mocks.retire.await_args.args[3]
    assert retained == {c["resource_name"] for c in calls}
    assert sync_mocks.record.await_args.kwargs["status"] == "ok"


@pytest.mark.asyncio
async def test_a_failed_folder_keeps_every_table(sync_mocks):
    sync_mocks.remember.side_effect = [
        SimpleNamespace(status="completed"),
        RuntimeError("boom"),
    ]

    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n", "id:w"]))

    sync_mocks.retire.assert_not_awaited()
    assert sync_mocks.record.await_args.kwargs["status"] == "degraded"


@pytest.mark.asyncio
async def test_an_empty_selection_syncs_nothing_and_retires_everything(sync_mocks):
    await DropboxIntegration().sync_now(credential(selected_folder_ids=[]))

    sync_mocks.source.assert_not_called()
    assert sync_mocks.retire.await_args.args[3] == set()


@pytest.mark.asyncio
async def test_a_folder_inside_another_selected_folder_is_refused(sync_mocks):
    with pytest.raises(sync_module.NestedSelectionError):
        await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n", "id:sub"]))

    sync_mocks.source.assert_not_called()
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.parametrize("selection", [None, "id:n", ["/Notes"], [""], ["id:"], ["id:a b"]])
@pytest.mark.asyncio
async def test_a_selection_that_is_not_folder_ids_is_refused(sync_mocks, selection):
    with pytest.raises(ValueError, match="folder ids"):
        await DropboxIntegration().sync_now(credential(selected_folder_ids=selection))
    sync_mocks.retire.assert_not_awaited()


def test_failures_are_classified_from_the_sources_typed_errors():
    from cognee.tasks.ingestion.connectors.dropbox import (
        DropboxAuthError,
        DropboxRateLimitedError,
    )

    wrapped = RuntimeError("pipeline failed")
    wrapped.__cause__ = DropboxAuthError("x")
    assert sync_module._classify_error(wrapped) == "failed_auth"
    assert sync_module._classify_error(DropboxRateLimitedError("x")) == "failed_rate_limit"
    from cognee.tasks.ingestion.connectors.dropbox import DropboxAccountMismatchError

    assert sync_module._classify_error(DropboxAccountMismatchError("x")) == "failed_account"
    assert sync_module._classify_error(ValueError()) == "failed_ingestion"


@pytest.mark.asyncio
async def test_a_scheduled_run_reaches_the_dropbox_sync_now():
    import cognee.modules.integrations.dropbox
    from cognee.modules.integrations import scheduled_sync

    row = SimpleNamespace(status="active", provider="dropbox")
    session = AsyncMock()
    session.get = AsyncMock(return_value=row)
    engine = Mock()
    engine.get_async_session.return_value.__aenter__ = AsyncMock(return_value=session)
    engine.get_async_session.return_value.__aexit__ = AsyncMock(return_value=False)
    with (
        patch.object(scheduled_sync, "get_relational_engine", return_value=engine),
        patch.object(DropboxIntegration, "sync_now", AsyncMock()) as sync_now,
    ):
        await scheduled_sync._sync(UUID(int=1))
    sync_now.assert_awaited_once_with(row)


@pytest.mark.asyncio
async def test_disconnect_revokes_the_token_and_survives_a_failed_revoke():
    with (
        patch.object(
            adapter_module, "decrypt_token_payload", return_value={"access_token": "sl.a"}
        ),
        patch.object(client_module, "revoke_token", AsyncMock()) as revoke,
    ):
        await DropboxIntegration().revoke_remote(credential())
    revoke.assert_awaited_once_with("sl.a")

    with (
        patch.object(
            adapter_module, "decrypt_token_payload", return_value={"access_token": "sl.a"}
        ),
        patch.object(
            client_module,
            "revoke_token",
            AsyncMock(
                side_effect=client_module.DropboxIntegrationError("token revoke", "http_401")
            ),
        ),
    ):
        await DropboxIntegration().revoke_remote(credential())


def test_disconnect_with_delete_names_the_accounts_dataset():
    name = DropboxIntegration().dataset_name(credential())
    assert name.startswith("dropbox_person_example_com_")


@pytest.mark.asyncio
async def test_a_refresh_refused_with_invalid_grant_revokes_the_local_credential():
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"access_token": "a", "refresh_token": "r"},
        ),
        patch.object(
            client_module,
            "refresh_access_token",
            AsyncMock(
                side_effect=client_module.DropboxIntegrationError("token refresh", "invalid_grant")
            ),
        ),
        patch.object(adapter_module, "revoke_credential_if_current", AsyncMock()) as revoke,
        pytest.raises(client_module.DropboxIntegrationError),
    ):
        await DropboxIntegration().refresh(credential())
    revoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_refresh_keeps_the_refresh_token():
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"access_token": "a", "refresh_token": "r"},
        ),
        patch.object(
            client_module,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "b", "expires_in": 14400}),
        ),
        patch.object(adapter_module, "update_refreshed_credential", AsyncMock()) as update,
    ):
        await DropboxIntegration().refresh(credential())
    assert update.await_args.kwargs["token_payload"] == {"access_token": "b", "refresh_token": "r"}


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"error": "invalid_grant", "error_description": "secret text"}, 400, "invalid_grant"),
        ({"error": {".tag": "path", "path": {".tag": "not_found"}}}, 409, "path"),
        (None, 500, "http_500"),
        ({"error": "Some Free Text"}, 400, "http_400"),
    ],
)
def test_client_errors_carry_a_tag_or_a_status_only(body, status, code):
    assert client_module._code(body, status) == code


@pytest.mark.asyncio
async def test_each_folder_asks_the_credential_store_for_a_fresh_token(sync_mocks):
    """A sync of several folders can outlast a four-hour token: no folder keeps the first."""
    import asyncio

    sync_mocks.token.side_effect = ["at-start", "renewed-1", "renewed-2"]

    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n", "id:w"]))

    second = sync_mocks.source.call_args_list[1].kwargs["token"]
    first = sync_mocks.source.call_args_list[0].kwargs["token"]
    # The source's client calls it on dlt's worker thread, before each request.
    assert await asyncio.to_thread(second) == "renewed-1"
    assert await asyncio.to_thread(first) == "renewed-2"


@pytest.mark.asyncio
async def test_a_nested_selection_is_refused_when_it_is_saved():
    folders = {"id:n": "/notes", "id:sub": "/notes/sub", "id:w": "/work"}
    lookup = AsyncMock(side_effect=lambda _t, f: {".tag": "folder", "path_lower": folders[f]})
    with (
        patch.object(adapter_module, "access_token_for", AsyncMock(return_value="t")),
        patch.object(sync_module.client, "get_folder", lookup),
        patch.object(
            sync_module.ingestion,
            "require_active_credential",
            AsyncMock(side_effect=lambda c: c),
        ),
    ):
        await DropboxIntegration().validate_resource_selection(credential(), ["id:n", "id:w"])
        with pytest.raises(sync_module.NestedSelectionError):
            await DropboxIntegration().validate_resource_selection(credential(), ["id:n", "id:sub"])
        with pytest.raises(ValueError, match="folder ids"):
            await DropboxIntegration().validate_resource_selection(credential(), ["/Notes"])


@pytest.mark.asyncio
async def test_a_paused_connection_is_not_started_by_the_schedule_and_resume_starts_it(
    database,
):
    """Through the sync policy store and tick (SDK-1434) on a real SQLite file: the tick's
    start is what the scheduler does, a call of the connection's ``sync_now``."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from cognee.modules.integrations.sync_policy import SyncPolicy

    store = import_module("cognee.modules.integrations.sync_policy.store")
    tick = import_module("cognee.modules.integrations.sync_policy.tick")
    saved = datetime(2026, 11, 5, 14, 0, tzinfo=timezone.utc)
    now = saved + timedelta(days=2)
    owner = SimpleNamespace(id=uuid4())
    connection = uuid4()
    await store.save_sync_policy(
        owner, owner.id, SyncPolicy(interval_minutes=60), connection, now=saved
    )
    await store.pause_sync_policy(owner, owner.id, connection, now=saved)

    async def owned(_principal_id):
        return [connection]

    with patch.object(DropboxIntegration, "sync_now", AsyncMock()) as sync_now:

        async def start(credential_id):
            await DropboxIntegration().sync_now(SimpleNamespace(id=credential_id))

        assert await tick.run_due_syncs(now, owned, start) == 0
        sync_now.assert_not_awaited()

        await store.resume_sync_policy(owner, owner.id, connection, now=now)
        assert await tick.run_due_syncs(now + timedelta(hours=1), owned, start) == 1
        assert sync_now.await_args.args[0].id == connection


@pytest.mark.asyncio
async def test_disconnect_revokes_with_a_fresh_token_when_the_stored_one_expired():
    from datetime import datetime, timedelta, timezone

    expired = credential()
    expired.token_expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"access_token": "sl.expired", "refresh_token": "r"},
        ),
        patch.object(
            client_module,
            "refresh_access_token",
            AsyncMock(return_value={"access_token": "sl.fresh", "expires_in": 14400}),
        ) as refresh,
        patch.object(client_module, "revoke_token", AsyncMock()) as revoke,
        patch.object(adapter_module, "update_refreshed_credential", AsyncMock()) as write,
    ):
        await DropboxIntegration().revoke_remote(expired)

    refresh.assert_awaited_once()
    revoke.assert_awaited_once_with("sl.fresh")
    write.assert_not_awaited()


def test_a_refresh_failure_from_the_token_bridge_counts_as_auth():
    from cognee.modules.integrations.credentials import CredentialInactiveError

    wrapped = RuntimeError("pipeline step failed")
    wrapped.__cause__ = client_module.DropboxIntegrationError("token refresh", "invalid_grant")
    assert sync_module._classify_error(wrapped) == "failed_auth"
    assert sync_module._classify_error(CredentialInactiveError("x")) == "failed_auth"
    other = client_module.DropboxIntegrationError("token refresh", "http_500")
    assert sync_module._classify_error(other) == "failed_ingestion"


@pytest.mark.asyncio
async def test_a_token_that_expires_between_two_folders_is_renewed_for_the_second(
    credential_db, monkeypatch
):
    """A real credential row crosses its expiry during a two-folder sync. Only Dropbox's HTTP
    answers are faked (``client._post``): the refresh, its write to the row and the reads
    are the real ones."""
    import asyncio
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from cognee.modules.integrations import credentials as store

    start = datetime(2026, 11, 11, 12, 0, tzinfo=timezone.utc)
    clock = [start]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(adapter_module, "datetime", Clock)
    row = await store.upsert_credential(
        provider="dropbox",
        provider_account_id=ACCOUNT,
        user_id=uuid4(),
        token_payload={"access_token": "sl.first", "refresh_token": "refresh"},
        provider_metadata={"email": "p@example.com", "selected_folder_ids": ["id:n", "id:w"]},
        token_expires_at=start + timedelta(minutes=30),
    )
    refreshes = []

    async def dropbox_http(operation, url, **kwargs):
        if url == client_module.TOKEN_URL:
            refreshes.append(kwargs["data"]["refresh_token"])
            return {"access_token": "sl.renewed", "expires_in": 14400}
        if url == client_module.METADATA_URL:
            path = {"id:n": "/notes", "id:w": "/work"}[kwargs["json"]["path"]]
            return {".tag": "folder", "path_lower": path}
        raise AssertionError(url)

    monkeypatch.setattr(client_module, "_post", dropbox_http)
    tokens = []

    async def remember(source, **_kwargs):
        # The source's client asks for its token on dlt's worker thread before a request.
        tokens.append(await asyncio.to_thread(source.token))
        clock[0] += timedelta(hours=1)  # the first folder outlasts the token

    monkeypatch.setattr(import_module("cognee.api.v1.remember.remember"), "remember", remember)
    monkeypatch.setattr(
        sync_module.ingestion,
        "source_factory",
        lambda _provider: lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(sync_module.ingestion, "retire_resources", AsyncMock())
    monkeypatch.setattr(
        "cognee.modules.integrations.sync_progress.report_sync_progress", AsyncMock()
    )
    monkeypatch.setattr("cognee.modules.users.methods.get_user", AsyncMock(return_value="owner"))
    monkeypatch.setattr(adapter_module, "require", lambda _name: "configured")

    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    status, counts = await sync_module._sync_source(row, counts)

    assert tokens == ["sl.first", "sl.renewed"] and refreshes == ["refresh"]
    assert status == "ok" and counts["failed"] == 0
    stored = await store.require_active_credential(row)
    assert store.decrypt_token_payload(stored) == {
        "access_token": "sl.renewed",
        "refresh_token": "refresh",
    }
