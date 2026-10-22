"""Slack selection and native-ingestion boundaries; no live Slack/LLM calls."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.modules.integrations.slack import history
from cognee.modules.integrations.slack.history_client import SlackAPIError
from cognee.modules.integrations.slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    parse_thread_link,
)

OLD = "1700000000.000001"
RECENT = "1701000000.000002"
LATER = "1702000000.000003"


def request(dataset_id=None, **kwargs):
    return SlackHistoryRequest(
        dataset_id=dataset_id or uuid4(),
        **{
            "channel_ids": ["C1"],
            "oldest": datetime.fromtimestamp(1700500000, timezone.utc),
            "latest": datetime.fromtimestamp(1701500000, timezone.utc),
            **kwargs,
        },
    )


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"channel_ids": ["C1"]},
        {"channel_ids": ["D1"], "days": 7},
        {"channel_ids": ["#general"], "days": 7},
        {"channel_ids": ["C1"], "days": 0},
        {"channel_ids": ["C1"], "days": -1},
        {"channel_ids": ["C1"], "days": 7, "oldest": "2026-01-01T00:00:00Z"},
        {"channel_ids": ["C1"], "oldest": "2026-01-01T00:00:00"},
        {"channel_ids": ["C1"], "oldest": "2026-02-01T00:00:00Z", "latest": "2026-01-01T00:00:00Z"},
    ],
)
def test_invalid_selection_is_rejected(values):
    with pytest.raises(ValidationError):
        SlackHistoryRequest(dataset_id=uuid4(), **values)


@pytest.mark.parametrize(
    "link",
    [
        "http://acme.slack.com/archives/C1/p1700000000000001",
        "https://slack.com.evil.test/archives/C1/p1700000000000001",
        "https://acme.slack.com@evil.test/archives/C1/p1700000000000001",
        "file:///etc/passwd",
        "https://127.0.0.1/archives/C1/p1700000000000001",
        "https://acme.slack.com/archives/D1/p1700000000000001",
        "https://acme.slack.com/archives/C1/p1700000000000001?thread_ts=oops",
    ],
)
def test_thread_links_are_not_arbitrary_urls(link):
    with pytest.raises(ValueError):
        parse_thread_link(link)


def test_reply_link_uses_explicit_parent_without_float_rounding():
    assert parse_thread_link(
        f"https://acme.slack.com/archives/C1/p1702000000000003?thread_ts={OLD}"
    ) == ("C1", OLD)


def test_window_resolves_days_against_one_instant():
    now = datetime(2026, 9, 6, tzinfo=timezone.utc)
    selection = SlackHistoryRequest(dataset_id=uuid4(), channel_ids=["C1", "C1"], days=7)
    oldest, latest = selection.bounds(now)
    assert oldest == datetime(2026, 8, 30, tzinfo=timezone.utc)
    assert latest == now
    assert selection.channel_ids == ["C1"]


@pytest.fixture
def harness(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    credential = SimpleNamespace(
        provider_account_id="T1",
        id=uuid4(),
        user_id=user.id,
        status="active",
        provider_metadata={"installed_by_slack_user_id": "U1", "bot_user_id": "UBOT"},
    )
    rows = {}
    client = SimpleNamespace(
        call=AsyncMock(return_value={"team_id": "T1", "url": "https://acme.slack.com/"}),
        authorize_channel=AsyncMock(
            side_effect=lambda channel, user: {"id": channel, "name": "general"}
        ),
        history=AsyncMock(return_value=[{"ts": RECENT}]),
        thread=AsyncMock(return_value=[{"ts": RECENT, "user": "U2", "text": "Ship Monday"}]),
    )

    class ClientContext:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            pass

    async def save(item, **kwargs):
        stamp = item.system_metadata[history.SOURCE]
        rows[(stamp["channel_id"], stamp["thread_ts"])] = SimpleNamespace(
            id=item.data_id,
            system_metadata=item.system_metadata,
            content=item.data,
            dataset_id=kwargs["dataset_id"],
        )

    async def change(data_id, item, **kwargs):
        assert item.data_id == data_id
        await save(item, **kwargs)

    async def delete(**kwargs):
        for key in list(rows):
            if rows[key].id == kwargs["data_id"]:
                rows.pop(key)

    add = AsyncMock(side_effect=save)
    update = AsyncMock(side_effect=change)
    remove = AsyncMock(side_effect=delete)
    cognify = AsyncMock()
    permission = AsyncMock()
    monkeypatch.setattr(history, "get_by_team", AsyncMock(return_value=credential))
    monkeypatch.setattr(history, "check_permission_on_dataset", permission)
    monkeypatch.setattr(
        history, "decrypt_token_payload", lambda credential: {"access_token": "fake"}
    )
    monkeypatch.setattr(history, "SlackHistoryClient", ClientContext)
    monkeypatch.setattr(
        history, "_stored_documents", AsyncMock(side_effect=lambda *args: dict(rows))
    )
    monkeypatch.setattr(history, "add", add)
    monkeypatch.setattr(history, "update", update)
    monkeypatch.setattr(history.datasets, "delete_data", remove)
    monkeypatch.setattr(history, "cognify", cognify)
    return SimpleNamespace(
        user=user,
        credential=credential,
        rows=rows,
        client=client,
        add=add,
        update=update,
        remove=remove,
        cognify=cognify,
        permission=permission,
    )


@pytest.mark.asyncio
async def test_repeat_import_reuses_document_and_edits_use_native_update(harness):
    selection = request()
    first = await history.import_slack_history("T1", selection, user=harness.user)
    row = harness.rows[("C1", RECENT)]
    assert first.added == 1
    assert "Ship Monday" in row.content
    assert "<@U2>" in row.content and "https://acme.slack.com/archives/C1/" in row.content
    again = await history.import_slack_history("T1", selection, user=harness.user)
    assert again.unchanged == 1
    harness.add.assert_awaited_once()
    harness.client.thread.return_value[0]["text"] = "Ship Tuesday"
    changed = await history.import_slack_history("T1", selection, user=harness.user)
    assert changed.updated == 1
    assert harness.rows[("C1", RECENT)].id == row.id
    assert "Ship Monday" not in harness.rows[("C1", RECENT)].content
    assert harness.update.call_args.kwargs["chunk_level_diff"] is False
    assert harness.add.call_args.kwargs["node_set"] == ["slack", "slack:T1", "slack:channel:C1"]


@pytest.mark.asyncio
async def test_shorter_window_never_deletes_older_memories(harness):
    selection = request()
    await history.import_slack_history("T1", selection, user=harness.user)
    harness.client.history.return_value = []
    result = await history.import_slack_history("T1", selection, user=harness.user)
    assert result.deleted == 0
    assert len(harness.rows) == 1
    harness.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_complete_thread_keeps_context_outside_date_window(harness):
    harness.client.thread.return_value.append(
        {"ts": LATER, "thread_ts": RECENT, "user": "U3", "text": "Agreed"}
    )
    result = await history.import_slack_history("T1", request(), user=harness.user)
    assert result.messages == 2
    assert "Agreed" in harness.rows[("C1", RECENT)].content


@pytest.mark.asyncio
async def test_active_window_finds_new_reply_to_old_root(harness):
    harness.client.history.return_value = [{"ts": OLD, "reply_count": 1}]
    harness.client.thread.return_value = [
        {"ts": OLD, "user": "U2", "text": "Old question"},
        {"ts": RECENT, "thread_ts": OLD, "user": "U3", "text": "New answer"},
    ]
    result = await history.import_slack_history(
        "T1", request(thread_mode="active"), user=harness.user
    )
    assert result.conversations == 1 and result.messages == 2
    assert harness.client.history.call_args.kwargs["oldest"] == "0"
    assert ("C1", OLD) in harness.rows


@pytest.mark.asyncio
async def test_active_window_ignores_inactive_old_threads(harness):
    harness.client.history.return_value = [{"ts": OLD, "reply_count": 1}]
    harness.client.thread.return_value = [{"ts": OLD, "user": "U2", "text": "Old question"}]
    result = await history.import_slack_history(
        "T1", request(thread_mode="active"), user=harness.user
    )
    assert result.conversations == 0
    harness.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_thread_links_work_without_channel_window(harness):
    selection = SlackHistoryRequest(
        dataset_id=uuid4(),
        thread_links=[f"https://acme.slack.com/archives/C1/p{RECENT.replace('.', '')}"],
    )
    await history.import_slack_history("T1", selection, user=harness.user)
    harness.client.history.assert_not_awaited()
    harness.client.authorize_channel.assert_awaited_once_with("C1", "U1")
    harness.client.thread.assert_awaited_once_with("C1", RECENT)


@pytest.mark.asyncio
async def test_other_workspace_link_is_refused(harness):
    selection = request(
        thread_links=[f"https://other.slack.com/archives/C1/p{RECENT.replace('.', '')}"]
    )
    with pytest.raises(SlackHistoryError, match="another Slack workspace"):
        await history.import_slack_history("T1", selection, user=harness.user)
    harness.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_any_fetch_failure_causes_no_native_writes(harness):
    harness.client.history.side_effect = [
        [{"ts": RECENT}],
        SlackAPIError("history", "missing_scope"),
    ]
    with pytest.raises(SlackHistoryError, match="missing_scope"):
        await history.import_slack_history(
            "T1", request(channel_ids=["C1", "C2"]), user=harness.user
        )
    harness.add.assert_not_awaited()
    harness.update.assert_not_awaited()
    harness.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_native_permission_denied_before_source_fetch(harness):
    harness.permission.side_effect = PermissionError("no grant")
    with pytest.raises(PermissionError):
        await history.import_slack_history("T1", request(), user=harness.user)
    harness.client.call.assert_not_awaited()
    harness.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_cognee_user_cannot_use_installation(harness):
    with pytest.raises(SlackHistoryError, match="owner"):
        await history.import_slack_history("T1", request(), user=SimpleNamespace(id=uuid4()))
    harness.client.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_target_channels_must_also_obey_allowlist(harness):
    harness.credential.provider_metadata["allowed_channel_ids"] = ["C9"]
    with pytest.raises(SlackHistoryError, match="allowlist"):
        await history.import_slack_history("T1", request(), user=harness.user)
    harness.client.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_detects_deleted_thread_without_window_cleanup(harness):
    selection = request()
    await history.import_slack_history("T1", selection, user=harness.user)
    original = harness.rows[("C1", RECENT)].id
    harness.client.history.return_value = []
    harness.client.thread.side_effect = SlackAPIError("replies", "thread_not_found")
    result = await history.import_slack_history("T1", selection, user=harness.user, reconcile=True)
    assert result.deleted == 1
    assert harness.remove.call_args.kwargs["data_id"] == original


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", ["missing_scope", "not_in_channel", "channel_not_found", "token_revoked"]
)
async def test_lost_source_access_is_never_interpreted_as_deletion(harness, code):
    selection = request()
    await history.import_slack_history("T1", selection, user=harness.user)
    harness.client.history.return_value = []
    harness.client.thread.side_effect = SlackAPIError("replies", code)
    with pytest.raises(SlackHistoryError):
        await history.import_slack_history("T1", selection, user=harness.user, reconcile=True)
    harness.remove.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_catches_deleted_reply_and_late_reply(harness):
    selection = request()
    harness.client.thread.return_value.append(
        {"ts": LATER, "thread_ts": RECENT, "user": "U3", "text": "Deleted reply"}
    )
    await history.import_slack_history("T1", selection, user=harness.user)
    harness.client.history.return_value = []
    harness.client.thread.return_value = [{"ts": RECENT, "text": "Parent only", "user": "U2"}]
    result = await history.import_slack_history("T1", selection, user=harness.user, reconcile=True)
    assert result.updated == 1
    assert "Deleted reply" not in harness.rows[("C1", RECENT)].content


@pytest.mark.asyncio
async def test_retry_runs_cognify_after_raw_data_was_saved(harness):
    selection = request()
    harness.cognify.side_effect = RuntimeError("indexing failed")
    with pytest.raises(RuntimeError):
        await history.import_slack_history("T1", selection, user=harness.user)
    harness.cognify.side_effect = None
    result = await history.import_slack_history("T1", selection, user=harness.user)
    assert result.unchanged == 1
    assert harness.cognify.await_count == 2
    assert harness.cognify.call_args.kwargs["raise_on_error"] is True


@pytest.mark.asyncio
async def test_permission_revoked_during_fetch_prevents_write(harness):
    harness.permission.side_effect = [None, None, None, PermissionError("revoked")]
    with pytest.raises(PermissionError):
        await history.import_slack_history("T1", request(), user=harness.user)
    harness.add.assert_not_awaited()


@pytest.mark.asyncio
async def test_cognee_bot_echoes_are_excluded(harness):
    harness.client.thread.return_value.append(
        {"ts": LATER, "thread_ts": RECENT, "user": "UBOT", "text": "Cognee answer"}
    )
    result = await history.import_slack_history("T1", request(), user=harness.user)
    assert result.messages == 1
    assert "Cognee answer" not in harness.rows[("C1", RECENT)].content
