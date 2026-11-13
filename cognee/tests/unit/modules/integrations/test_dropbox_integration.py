"""The Dropbox integration module: connect, picker, selection, Sync now and disconnect.

Provider calls are mocked at ``dropbox.client``; ingestion is mocked at ``ingestion``. Pause,
resume and schedule are the shared router's and the sync policies' (SDK-1434), tested there;
here only that a scheduled run reaches this adapter's ``sync_now``.
"""

import json
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
        ciphertext=b"stored",
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
                side_effect=lambda _t, folder_id, **_kw: {
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
async def test_the_cached_token_is_fetched_again_once_it_nears_expiry(sync_mocks):
    """A sync of several folders can outlast a four-hour token: the cache goes back to the
    credential store when the token it holds is within the refresh margin."""
    import asyncio
    from datetime import datetime, timedelta, timezone

    expired = datetime.now(timezone.utc) - timedelta(minutes=1)
    fetch = AsyncMock(side_effect=[("renewed-1", expired, b"1"), ("renewed-2", None, b"2")])
    with patch.object(adapter_module, "access_token_and_expiry", fetch):
        await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n", "id:w"]))
        token = sync_mocks.source.call_args_list[1].kwargs["token"]
        # The source's client calls it on dlt's worker thread, before each request.
        assert await asyncio.to_thread(token) == "renewed-1"
        assert await asyncio.to_thread(token) == "renewed-2"
        assert await asyncio.to_thread(token) == "renewed-2"
    assert fetch.await_count == 2


@pytest.mark.asyncio
async def test_a_nested_selection_is_refused_when_it_is_saved():
    folders = {"id:n": "/notes", "id:sub": "/notes/sub", "id:w": "/work"}
    lookup = AsyncMock(
        side_effect=lambda _t, f, **_kw: {".tag": "folder", "path_lower": folders[f]}
    )
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


# ---------------------------------------------------------------- PR #6668 review bot
async def _real_two_folder_sync(monkeypatch, *, requests_per_folder, expires_in):
    """A sync over a real credential row whose source makes ``requests_per_folder`` requests.

    Returns the number of credential reads the adapter made and the tokens the requests used.
    """
    import asyncio
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from cognee.modules.integrations import credentials as store

    row = await store.upsert_credential(
        provider="dropbox",
        provider_account_id=ACCOUNT,
        user_id=uuid4(),
        token_payload={"access_token": "sl.first", "refresh_token": "refresh"},
        provider_metadata={"email": "p@example.com", "selected_folder_ids": ["id:n", "id:w"]},
        token_expires_at=datetime.now(timezone.utc) + expires_in,
    )
    reads = []
    real_read = adapter_module.require_active_credential

    async def counted_read(credential):
        reads.append(1)
        return await real_read(credential)

    monkeypatch.setattr(adapter_module, "require_active_credential", counted_read)

    async def dropbox_http(operation, url, **kwargs):
        if url == client_module.METADATA_URL:
            path = {"id:n": "/notes", "id:w": "/work"}[kwargs["json"]["path"]]
            return {".tag": "folder", "path_lower": path}
        raise AssertionError(url)

    monkeypatch.setattr(client_module, "_post", dropbox_http)
    tokens = []

    async def remember(source, **_kwargs):
        for _ in range(requests_per_folder):
            tokens.append(await asyncio.to_thread(source.token))

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
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    await sync_module._sync_source(row, counts)
    return len(reads), tokens


@pytest.mark.asyncio
async def test_the_token_path_reads_the_credential_once_per_token_lifetime(
    credential_db, monkeypatch
):
    from datetime import timedelta

    reads, tokens = await _real_two_folder_sync(
        monkeypatch, requests_per_folder=25, expires_in=timedelta(hours=4)
    )

    assert tokens == ["sl.first"] * 50
    # The host's active check (check_active, not called by this test's source) still reads the
    # row before every request; this counts the token path alone.
    assert reads <= 2, f"{reads} credential reads for 50 requests"


def test_a_cached_token_is_replaced_after_a_401_and_the_request_retried_once():
    import httpx

    from cognee.modules.integrations.dropbox.sync import _TokenCache

    tokens = iter(["sl.stale", "sl.renewed"])
    cache = _TokenCache(lambda _refused: (next(tokens), None, b"row"))
    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        if request.headers["Authorization"] == "Bearer sl.stale":
            return httpx.Response(401, json={"error": {".tag": "expired_access_token"}})
        return httpx.Response(200, json={"ok": True})

    from cognee.tasks.ingestion.connectors.dropbox import DropboxClient

    client = DropboxClient(cache, http=httpx.Client(transport=httpx.MockTransport(handler)))
    assert client.rpc("files/get_metadata", {"path": "id:x"}) == {"ok": True}
    assert seen == ["Bearer sl.stale", "Bearer sl.renewed"]


@pytest.mark.asyncio
async def test_the_file_size_setting_reaches_every_source(sync_mocks, monkeypatch):
    monkeypatch.setattr(dropbox_settings, "max_file_size_mb", 5.0, raising=False)
    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n"]))
    assert sync_mocks.source.call_args.kwargs["max_file_size_mb"] == 5.0

    monkeypatch.setattr(dropbox_settings, "max_file_size_mb", 200.0, raising=False)
    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:w"]))
    assert sync_mocks.source.call_args.kwargs["max_file_size_mb"] == 200.0


def test_the_file_size_setting_comes_from_the_environment(monkeypatch):
    from cognee.modules.integrations.dropbox.dropbox_settings import DropboxSettings

    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "2.5")
    assert DropboxSettings().max_file_size_mb == 2.5
    monkeypatch.delenv("DROPBOX_MAX_FILE_SIZE_MB")
    assert DropboxSettings().max_file_size_mb == 50, "the integration's default cap"


@pytest.mark.asyncio
async def test_the_folder_picker_reuses_one_session_across_pages_and_keeps_folders_only(
    monkeypatch,
):
    pages = iter(
        [
            {
                "entries": [
                    {".tag": "folder", "id": "id:a", "path_display": "/A"},
                    {".tag": "file", "id": "id:f", "path_display": "/A/f.md"},
                ],
                "cursor": "c1",
                "has_more": True,
            },
            {
                "entries": [{".tag": "folder", "id": "id:b", "path_display": "/B"}],
                "cursor": "c2",
                "has_more": False,
            },
        ]
    )
    sessions = []

    class Response:
        status = 200

        async def json(self, content_type=None):
            return next(pages)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def __init__(self, **_kwargs):
            sessions.append(self)

        def post(self, *_args, **_kwargs):
            return Response()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(client_module, "client_session", lambda _provider, **kw: Session(**kw))

    folders = await client_module.list_folders("sl.token")

    assert [f["id"] for f in folders] == ["id:a", "id:b"]
    assert len(sessions) == 1, f"{len(sessions)} HTTP sessions for two pages"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["notion", "dropbox"])
async def test_notion_and_dropbox_renew_tokens_through_one_shared_helper(provider):
    from cognee.modules.integrations import credentials

    adapter = import_module(f"cognee.modules.integrations.{provider}.adapter")
    row = credential()
    with (
        patch.object(credentials, "refreshed_credential", AsyncMock(return_value=row)) as shared,
        patch.object(adapter, "decrypt_token_payload", return_value={"access_token": "tok"}),
    ):
        assert await adapter.access_token_for(row) == "tok"
    shared.assert_awaited_once()


# ---------------------------------------------------------------- review swarm round 8
@pytest.mark.parametrize("value", ["", " "])
def test_an_empty_file_size_setting_means_the_default(monkeypatch, value):
    from cognee.modules.integrations.dropbox.dropbox_settings import DropboxSettings

    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", value)
    assert DropboxSettings().max_file_size_mb == 50


@pytest.mark.parametrize("value", ["inf", "nan", "0", "-1"])
def test_a_file_size_setting_that_would_skip_everything_or_nothing_is_refused(monkeypatch, value):
    from pydantic import ValidationError

    from cognee.modules.integrations.dropbox.dropbox_settings import DropboxSettings

    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", value)
    with pytest.raises(ValidationError):
        DropboxSettings()


def test_the_api_imports_with_an_empty_file_size_setting():
    import os
    import subprocess
    import sys

    env = {**os.environ, "DROPBOX_MAX_FILE_SIZE_MB": ""}
    subprocess.run(
        [sys.executable, "-c", "import cognee.modules.integrations.dropbox"], env=env, check=True
    )


async def _token_source_of_a_real_sync(monkeypatch, during):
    """Run a one-folder sync over a real credential row; ``during(token, row)`` runs inside
    the source's walk on dlt's worker thread and returns what the test wants to see."""
    import asyncio
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from cognee.modules.integrations import credentials as store

    row = await store.upsert_credential(
        provider="dropbox",
        provider_account_id=ACCOUNT,
        user_id=uuid4(),
        token_payload={"access_token": "sl.first", "refresh_token": "refresh"},
        provider_metadata={"email": "p@example.com", "selected_folder_ids": ["id:n"]},
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=4),
    )
    refreshes = []

    async def dropbox_http(operation, url, **kwargs):
        if url == client_module.TOKEN_URL:
            refreshes.append(1)
            return {"access_token": "sl.renewed", "expires_in": 14400}
        if url == client_module.METADATA_URL:
            return {".tag": "folder", "path_lower": "/notes"}
        raise AssertionError(url)

    monkeypatch.setattr(client_module, "_post", dropbox_http)
    monkeypatch.setattr(adapter_module, "require", lambda _name: "configured")
    seen = []
    loop = asyncio.get_running_loop()

    async def remember(source, **_kwargs):
        def walk():
            return during(source.token, row, loop)

        seen.append(await asyncio.to_thread(walk))

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
    await sync_module._sync_source(row, {"scanned": 0, "skipped": 0, "failed": 0})
    return seen[0], refreshes


@pytest.mark.asyncio
async def test_a_401_on_a_token_the_store_thinks_fresh_refreshes_it(credential_db, monkeypatch):
    def during(token, _row, _loop):
        first = token()
        token.invalidate()  # what the client does when Dropbox answers 401
        return first, token()

    (first, retried), refreshes = await _token_source_of_a_real_sync(monkeypatch, during)

    assert first == "sl.first" and retried == "sl.renewed" and refreshes == [1]


@pytest.mark.asyncio
async def test_a_401_after_a_reconnect_takes_the_new_token_without_a_refresh(
    credential_db, monkeypatch
):
    import asyncio

    from cognee.modules.integrations import credentials as store

    def during(token, row, loop):
        first = token()
        # The owner reconnects the same account while the sync runs; disconnect revoked the
        # old token at Dropbox, so the next request answers 401.
        asyncio.run_coroutine_threadsafe(
            store.upsert_credential(
                provider="dropbox",
                provider_account_id=ACCOUNT,
                user_id=row.user_id,
                token_payload={"access_token": "sl.reconnected", "refresh_token": "r2"},
            ),
            loop,
        ).result()
        token.invalidate()
        return first, token()

    (first, retried), refreshes = await _token_source_of_a_real_sync(monkeypatch, during)

    assert first == "sl.first" and retried == "sl.reconnected" and refreshes == []


@pytest.mark.asyncio
async def test_a_refresh_answer_without_a_lifetime_is_refused():
    with (
        patch.object(
            adapter_module,
            "decrypt_token_payload",
            return_value={"access_token": "a", "refresh_token": "r"},
        ),
        patch.object(
            client_module, "refresh_access_token", AsyncMock(return_value={"access_token": "b"})
        ),
        patch.object(adapter_module, "update_refreshed_credential", AsyncMock()) as write,
        pytest.raises(client_module.DropboxIntegrationError, match="no_expires_in"),
    ):
        await DropboxIntegration().refresh(credential())
    write.assert_not_awaited()


def test_a_refresh_token_dropbox_refuses_is_an_auth_error_of_the_source():
    """invalid_grant: the adapter has revoked the connection; the source must see its own auth
    error, so the run stops cleanly with the rows it already has (failed_auth)."""
    from cognee.modules.integrations.dropbox.sync import _TokenCache
    from cognee.tasks.ingestion.connectors.dropbox import DropboxAuthError

    def fetch(_refused):
        raise client_module.DropboxIntegrationError("token refresh", "invalid_grant")

    with pytest.raises(DropboxAuthError, match="reconnect"):
        _TokenCache(fetch)()


@pytest.mark.asyncio
async def test_a_refresh_refused_mid_sync_revokes_the_connection_and_stops_with_failed_auth(
    credential_db, monkeypatch
):
    from cognee.modules.integrations import credentials as store
    from cognee.tasks.ingestion.connectors.dropbox import DropboxAuthError

    async def refused(*_args, **_kwargs):
        raise client_module.DropboxIntegrationError("token refresh", "invalid_grant")

    def during(token, _row, _loop):
        token()
        token.invalidate()  # Dropbox answered 401 for the token
        monkeypatch.setattr(client_module, "refresh_access_token", refused)
        try:
            token()
        except DropboxAuthError as error:
            return type(error).__name__
        return "no error"

    outcome, _ = await _token_source_of_a_real_sync(monkeypatch, during)

    assert outcome == "DropboxAuthError"
    persisted = await store.get_credential_by_account("dropbox", ACCOUNT)
    assert persisted.status == "revoked"


# ---------------------------------------------------------------- round 9 (bugs, red first)
@pytest.mark.asyncio
async def test_the_nested_selection_check_uses_one_session_and_one_credential_read(monkeypatch):
    folders = {f"id:f{i}": f"/folder{i}" for i in range(5)}
    sessions = []

    class Response:
        status = 200

        def __init__(self, body):
            self.body = body

        async def json(self, content_type=None):
            return self.body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class Session:
        def __init__(self, **_kwargs):
            sessions.append(self)

        def post(self, _url, **kwargs):
            return Response({".tag": "folder", "path_lower": folders[kwargs["json"]["path"]]})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    reads = AsyncMock(side_effect=lambda credential: credential)
    monkeypatch.setattr(client_module, "client_session", lambda _provider, **kw: Session(**kw))
    monkeypatch.setattr(sync_module.ingestion, "require_active_credential", reads)

    await sync_module.refuse_nested_selection(credential(), "sl.token", sorted(folders))

    assert len(sessions) == 1, f"{len(sessions)} HTTP sessions for 5 folders"
    assert reads.await_count == 1, f"{reads.await_count} credential reads for 5 folders"


# ---------------------------------------------------------------- round 10: retire after failures
async def _sync_folder_a(sync_mocks, monkeypatch, fake, folder_a):
    """Sync a selection of folder A only, the walk running for real over ``fake``; a folder B
    deselected earlier is what ``retire_resources`` would remove."""
    from cognee.tasks.ingestion.connectors.dropbox import (
        DropboxError,
        _iter_rows,
        _Session,
        dropbox_source,
    )

    fake.account = ACCOUNT
    walks = {}

    def build(**kwargs):
        source = dropbox_source(
            folder_id=kwargs["folder_id"],
            account_id=kwargs["account_id"],
            resource_name=kwargs["resource_name"],
            service=fake.client(sleep=lambda _s: None),
        )
        walks[id(source)] = kwargs["folder_id"]
        return source

    async def remember(source, **_kwargs):
        try:
            list(
                _iter_rows(
                    _Session(fake.client(sleep=lambda _s: None), None),
                    folder_id=walks[id(source)],
                    account_id=ACCOUNT,
                    state={},
                    stats=source.cognee_sync_stats,
                    max_bytes=None,
                )
            )
        except DropboxError as error:
            raise RuntimeError("pipeline step failed") from error
        return SimpleNamespace(status="completed")

    # Through the fixture's own mocks: patching over them here would outlive the fixture.
    sync_mocks.remember.side_effect = remember
    sync_mocks.source.side_effect = build
    sync_mocks.folder.side_effect = None
    sync_mocks.folder.return_value = {".tag": "folder", "path_lower": "/a"}
    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    status, counts = await sync_module._sync_source(
        credential(selected_folder_ids=[folder_a]), counts
    )
    return status, counts


@pytest.mark.parametrize("failing_files", [1, 3])
@pytest.mark.asyncio
async def test_files_that_keep_failing_do_not_stop_a_deselected_folder_from_retiring(
    sync_mocks, monkeypatch, failing_files
):
    from cognee.tests.unit.tasks.connectors.fake_dropbox import FakeDropbox

    fake = FakeDropbox()
    folder_a = fake.mkdir("/A")
    fake.put("/A/ok.md", "fine")
    for i in range(failing_files):
        fake.download_errors[fake.put(f"/A/broken{i}.md", "never")] = (503, {})

    status, counts = await _sync_folder_a(sync_mocks, monkeypatch, fake, folder_a)

    assert counts["failed_retry"] == failing_files and status == "degraded", counts
    sync_mocks.retire.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_folder_that_fails_while_listing_keeps_deselected_folders(sync_mocks, monkeypatch):
    from cognee.tests.unit.tasks.connectors.fake_dropbox import FakeDropbox

    fake = FakeDropbox()
    folder_a = fake.mkdir("/A")
    fake.put("/A/ok.md", "fine")
    fake.queued["files/list_folder"] = [(403, {})]

    status, counts = await _sync_folder_a(sync_mocks, monkeypatch, fake, folder_a)

    assert counts["failed_listing"] == 1 and status == "degraded"
    sync_mocks.retire.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_folder_cut_by_a_rate_limit_after_rows_lets_deselected_folders_retire(
    sync_mocks, monkeypatch
):
    from cognee.tests.unit.tasks.connectors.fake_dropbox import FakeDropbox

    fake = FakeDropbox()
    folder_a = fake.mkdir("/A")
    fake.put("/A/a.md", "first")
    fake.put("/A/b.md", "second")
    fake.queued["files/download"] = [(200, b"first"), (429, {}, {"Retry-After": "3600"})]

    status, counts = await _sync_folder_a(sync_mocks, monkeypatch, fake, folder_a)

    assert counts["failed_rate_limit"] == 1 and status == "degraded"
    sync_mocks.retire.assert_awaited_once()


# ---------------------------------------------------------------- round 11: a folder that is gone
@pytest.mark.asyncio
async def test_a_gone_folder_never_blocks_retiring_the_deselected_ones(sync_mocks):
    from cognee.tasks.ingestion.connectors.dropbox import (
        DropboxError,
        _iter_rows,
        _Session,
        dropbox_source,
    )
    from cognee.tests.unit.tasks.connectors.fake_dropbox import FakeDropbox

    fake = FakeDropbox()
    fake.account = ACCOUNT
    folder_a, folder_b = fake.mkdir("/A"), fake.mkdir("/B")
    fake.put("/A/a.md", "alpha")
    fake.put("/B/b.md", "bravo")
    states: dict[str, dict] = {}
    walks = {}

    def build(**kwargs):
        source = dropbox_source(
            folder_id=kwargs["folder_id"],
            account_id=kwargs["account_id"],
            resource_name=kwargs["resource_name"],
            service=fake.client(sleep=lambda _s: None),
        )
        walks[id(source)] = (kwargs["folder_id"], kwargs["resource_name"])
        return source

    async def remember(source, **_kwargs):
        folder_id, name = walks[id(source)]
        state = states.setdefault(name, {})
        try:
            list(
                _iter_rows(
                    _Session(fake.client(sleep=lambda _s: None), None),
                    folder_id=folder_id,
                    account_id=ACCOUNT,
                    state=state,
                    stats=source.cognee_sync_stats,
                    max_bytes=None,
                )
            )
        except DropboxError as error:
            raise RuntimeError("pipeline step failed") from error
        return SimpleNamespace(status="completed")

    async def get_folder(_token, folder_id, **_kwargs):
        meta = next((m for m in fake.folders.values() if m["id"] == folder_id), None)
        return {".tag": "folder", "path_lower": meta["path_lower"]} if meta else None

    sync_mocks.remember.side_effect = remember
    sync_mocks.source.side_effect = build
    sync_mocks.folder.side_effect = get_folder

    async def run(selection):
        sync_mocks.retire.reset_mock()
        counts = {"scanned": 0, "skipped": 0, "failed": 0}
        return await sync_module._sync_source(credential(selected_folder_ids=selection), counts)

    await run([folder_a, folder_b])
    fake.delete("/A")
    _, forgetting = await run([folder_a, folder_b])
    _, later = await run([folder_a, folder_b])
    status, deselected = await run([folder_a])

    assert forgetting["deleted"] == 1 and forgetting["failed"] == 0
    assert later.get("failed_listing", 0) == 0, later
    assert later["failed_folder_gone"] == 1 and deselected["failed_folder_gone"] == 1
    assert status == "degraded"
    sync_mocks.retire.assert_awaited_once()


# ---------------------------------------------------------------- round 12 (review)
def test_the_integration_caps_file_size_at_50_mb_by_default(monkeypatch):
    from cognee.modules.integrations.dropbox.dropbox_settings import DropboxSettings

    monkeypatch.delenv("DROPBOX_MAX_FILE_SIZE_MB", raising=False)
    assert DropboxSettings().max_file_size_mb == 50
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "")
    assert DropboxSettings().max_file_size_mb == 50, "blank means the default, not no limit"
    monkeypatch.setenv("DROPBOX_MAX_FILE_SIZE_MB", "200")
    assert DropboxSettings().max_file_size_mb == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [["id:gone"], ["id:n", "id:gone"]])
async def test_a_folder_id_dropbox_cannot_resolve_is_refused_when_saved(selection):
    async def get_folder(_token, folder_id, **_kwargs):
        return {".tag": "folder", "path_lower": "/notes"} if folder_id == "id:n" else None

    with (
        patch.object(adapter_module, "access_token_for", AsyncMock(return_value="t")),
        patch.object(sync_module.client, "get_folder", get_folder),
        patch.object(
            sync_module.ingestion, "require_active_credential", AsyncMock(side_effect=lambda c: c)
        ),
        pytest.raises(ValueError, match="could not be found"),
    ):
        await DropboxIntegration().validate_resource_selection(credential(), selection)


@pytest.mark.asyncio
async def test_an_unmounted_folder_without_a_path_does_not_count_as_a_parent(monkeypatch):
    metas = {"id:a": {".tag": "folder"}, "id:b": {".tag": "folder", "path_lower": "/b"}}

    async def get_folders(_token, folder_ids):
        return [metas[f] for f in folder_ids]

    monkeypatch.setattr(sync_module.client, "get_folders", get_folders)
    monkeypatch.setattr(
        sync_module.ingestion, "require_active_credential", AsyncMock(side_effect=lambda c: c)
    )

    await sync_module.refuse_nested_selection(credential(), "t", ["id:a", "id:b"])


@pytest.mark.asyncio
async def test_the_picker_lists_one_level_under_the_parent_it_is_given(monkeypatch):
    listed = []

    async def list_folders(_token, parent=None):
        listed.append(parent)
        return [{".tag": "folder", "id": "id:c", "path_lower": "/a/c", "path_display": "/A/c"}]

    monkeypatch.setattr(client_module, "list_folders", list_folders)
    monkeypatch.setattr(adapter_module, "access_token_for", AsyncMock(return_value="t"))

    top = await DropboxIntegration().list_resources(credential())
    child = await DropboxIntegration().list_resources(credential(), parent="id:a")

    assert listed == [None, "id:a"]
    assert [r["id"] for r in child] == ["id:c"] and top[0]["attributes"]["path"] == "/A/c"


@pytest.mark.asyncio
async def test_the_folder_listing_is_one_level_and_folders_only(monkeypatch):
    bodies = []

    async def post(operation, url, **kwargs):
        bodies.append(kwargs.get("json"))
        return {
            "entries": [
                {".tag": "folder", "id": "id:c", "path_display": "/A/c"},
                {".tag": "file", "id": "id:f", "path_display": "/A/f.md"},
            ],
            "cursor": "c1",
            "has_more": False,
        }

    monkeypatch.setattr(client_module, "_post", post)

    folders = await client_module.list_folders("t", parent="id:a")

    assert [f["id"] for f in folders] == ["id:c"]
    assert bodies[0]["recursive"] is False and bodies[0]["path"] == "id:a"


# ---------------------------------------------------------------- round 13
@pytest.mark.asyncio
async def test_a_folder_cut_by_its_time_budget_before_working_a_file_still_counts_as_cut(
    sync_mocks,
):
    """Only identical files (no download) before the budget ran out: scanned stays 0, yet the
    folder's state advanced and its table holds only its own rows."""

    def build(**kwargs):
        stats = dict.fromkeys(("scanned", "deleted", "failed", "failed_budget", "complete"), 0)
        stats.update(failed=1, failed_budget=1)
        return SimpleNamespace(cognee_sync_stats=stats, **kwargs)

    sync_mocks.source.side_effect = build

    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n"]))

    sync_mocks.retire.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_folder_without_a_path_is_refused_when_saved():
    async def get_folder(_token, folder_id, **_kwargs):
        return {".tag": "folder"} if folder_id == "id:unmounted" else None

    with (
        patch.object(adapter_module, "access_token_for", AsyncMock(return_value="t")),
        patch.object(sync_module.client, "get_folder", get_folder),
        patch.object(
            sync_module.ingestion, "require_active_credential", AsyncMock(side_effect=lambda c: c)
        ),
        pytest.raises(ValueError, match="could not be found"),
    ):
        await DropboxIntegration().validate_resource_selection(credential(), ["id:unmounted"])


def test_a_never_listed_missing_folder_has_its_own_key_that_never_blocks_retirement():
    from cognee.tasks.ingestion.connectors.dropbox import DropboxFolderNotFoundError, _failure_key

    error = DropboxFolderNotFoundError("x")
    assert _failure_key(error) == "failed_folder_not_found"
    wrapped = RuntimeError("pipeline step failed")
    wrapped.__cause__ = error
    assert sync_module._classify_error(wrapped) == "failed_folder_not_found"


# ---------------------------------------------------------------- the aiohttp client, real socket
class _DropboxOAuthServer:
    """A real HTTP server on a free local port that answers like Dropbox's OAuth and RPC routes."""

    def __init__(self, answers):
        import http.server
        import threading

        self.answers = answers
        self.requests = []
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.requests.append((self.path, dict(self.headers), body))
                status, payload = server.answers[self.path](body)
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.http = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self):
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def oauth_server(monkeypatch):
    answers: dict = {}
    server = _DropboxOAuthServer(answers)
    for name, path in {
        "TOKEN_URL": "/oauth2/token",
        "REVOKE_URL": "/2/auth/token/revoke",
        "ACCOUNT_URL": "/2/users/get_current_account",
        "METADATA_URL": "/2/files/get_metadata",
        "LIST_URL": "/2/files/list_folder",
        "CONTINUE_URL": "/2/files/list_folder/continue",
    }.items():
        monkeypatch.setattr(client_module, name, server.base + path)
    server.answers_for = answers
    yield server
    server.close()


@pytest.mark.asyncio
async def test_the_token_exchange_and_refresh_cross_a_real_socket(oauth_server):
    oauth_server.answers_for["/oauth2/token"] = lambda body: (
        200,
        {"access_token": "sl.a", "account_id": "dbid:A", "expires_in": 14400},
    )

    granted = await client_module.exchange_code(
        "code", client_id="key", client_secret="secret", redirect_uri="https://x/cb"
    )
    renewed = await client_module.refresh_access_token(
        "refresh", client_id="key", client_secret="secret"
    )

    assert granted["account_id"] == "dbid:A" and renewed["access_token"] == "sl.a"
    (_path, headers, body), second = oauth_server.requests
    assert b"grant_type=authorization_code" in body and headers["Authorization"].startswith(
        "Basic "
    )
    assert b"grant_type=refresh_token" in second[2]


@pytest.mark.asyncio
async def test_get_current_account_crosses_a_real_socket(oauth_server):
    oauth_server.answers_for["/2/users/get_current_account"] = lambda body: (
        200,
        {"account_id": "dbid:A", "email": "a@example.com"},
    )

    account = await client_module.get_current_account("sl.t")

    assert account["email"] == "a@example.com"
    assert oauth_server.requests[0][1]["Authorization"] == "Bearer sl.t"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected"),
    [
        (200, {".tag": "folder", "id": "id:a", "path_lower": "/a"}, "folder"),
        (200, {".tag": "file", "id": "id:f"}, None),
        (409, {"error": {".tag": "path", "path": {".tag": "not_found"}}}, None),
        (401, {"error": {".tag": "expired_access_token"}}, "expired_access_token"),
        (429, {"error": {".tag": "too_many_requests"}}, "too_many_requests"),
    ],
)
async def test_get_folder_crosses_a_real_socket(oauth_server, status, payload, expected):
    oauth_server.answers_for["/2/files/get_metadata"] = lambda body: (status, payload)

    if status in (401, 429):
        with pytest.raises(client_module.DropboxIntegrationError) as error:
            await client_module.get_folder("sl.t", "id:a")
        assert error.value.code == expected and "sl.t" not in str(error.value)
        return
    folder = await client_module.get_folder("sl.t", "id:a")
    assert (folder or {}).get(".tag") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [None, "id:a"])
async def test_list_folders_crosses_a_real_socket_with_a_continue_page(oauth_server, parent):
    oauth_server.answers_for["/2/files/list_folder"] = lambda body: (
        200,
        {"entries": [{".tag": "folder", "id": "id:1"}], "cursor": "c1", "has_more": True},
    )
    oauth_server.answers_for["/2/files/list_folder/continue"] = lambda body: (
        200,
        {
            "entries": [{".tag": "folder", "id": "id:2"}, {".tag": "file", "id": "id:f"}],
            "cursor": "c2",
            "has_more": False,
        },
    )

    folders = await client_module.list_folders("sl.t", parent=parent)

    assert [f["id"] for f in folders] == ["id:1", "id:2"]
    first = json.loads(oauth_server.requests[0][2])
    assert first == {"path": parent or "", "recursive": False, "limit": 2000}
    assert json.loads(oauth_server.requests[1][2]) == {"cursor": "c1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [b"null", b""])
async def test_revoke_crosses_a_real_socket_and_a_void_answer_is_success(oauth_server, payload):
    """auth/token/revoke is a void RPC: Dropbox answers 200 with a null (or empty) body."""
    oauth_server.answers_for["/2/auth/token/revoke"] = lambda body: (200, payload)

    await client_module.revoke_token("sl.t")

    assert oauth_server.requests[0][1]["Authorization"] == "Bearer sl.t"


@pytest.mark.asyncio
async def test_a_selected_folder_never_found_does_not_block_retiring_deselected_ones(
    sync_mocks, monkeypatch
):
    from cognee.tests.unit.tasks.connectors.fake_dropbox import FakeDropbox

    status, counts = await _sync_folder_a(sync_mocks, monkeypatch, FakeDropbox(), "id:missing")

    assert counts["failed_folder_not_found"] == 1 and status == "degraded"
    sync_mocks.retire.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_single_folder_sync_makes_no_nesting_lookup(monkeypatch):
    """One folder cannot sit inside another selected one: a sync checks the credential and
    asks Dropbox nothing."""
    lookup = AsyncMock(return_value=[{".tag": "folder", "path_lower": "/a"}])
    reads = AsyncMock(side_effect=lambda credential: credential)
    monkeypatch.setattr(sync_module.client, "get_folders", lookup)
    monkeypatch.setattr(sync_module.ingestion, "require_active_credential", reads)

    await sync_module.refuse_nested_selection(credential(), "sl.token", ["id:a"])

    lookup.assert_not_awaited()
    reads.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_refused_token_on_one_folder_is_counted_and_on_two_folders_raises(sync_mocks):
    """One folder asks Dropbox nothing before its source runs, so a token Dropbox refuses
    reaches the source and is counted as failed_auth; two or more folders meet it in the
    nesting check, which raises before any source runs."""
    from cognee.tasks.ingestion.connectors.dropbox import DropboxAuthError

    sync_mocks.folder.side_effect = client_module.DropboxIntegrationError(
        "folder lookup", "http_401"
    )
    sync_mocks.remember.side_effect = DropboxAuthError("refused")

    await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n"]))

    sync_mocks.folder.assert_not_awaited()
    record = sync_mocks.record.await_args.kwargs
    assert record["status"] == "degraded" and record["counts"]["failed_auth"] == 1

    sync_mocks.remember.reset_mock()
    with pytest.raises(client_module.DropboxIntegrationError):
        await DropboxIntegration().sync_now(credential(selected_folder_ids=["id:n", "id:w"]))
    sync_mocks.remember.assert_not_awaited()
