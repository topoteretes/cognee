import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.slack.routers.get_slack_history_router import get_slack_history_router
from cognee.modules.integrations.slack import handle_history, history_sync
from cognee.modules.integrations.slack.history_models import (
    SlackHistoryError,
    SlackHistoryRequest,
    SlackHistoryResult,
    SlackSyncSettings,
)
from cognee.modules.users.methods import get_authenticated_user


def selection():
    return SlackHistoryRequest(dataset_id=uuid4(), channel_ids=["C1"], days=7)


@pytest.mark.asyncio
async def test_config_freezes_start_and_preserves_six_hour_interval(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    patch = AsyncMock()
    monkeypatch.setattr(history_sync, "authorize_import", AsyncMock())
    monkeypatch.setattr(history_sync, "_patch_entry", patch)
    saved = await history_sync.configure_slack_sync(
        "T1",
        SlackSyncSettings(enabled=True, selection=selection()),
        user=user,
    )
    assert saved.selection.days is None
    assert saved.selection.oldest.tzinfo is not None
    assert saved.interval_seconds == 21600
    assert patch.call_args.args[1] == "history_sync"
    assert patch.call_args.kwargs["user_id"] == user.id


@pytest.mark.asyncio
async def test_unauthorized_request_cannot_write_another_connections_status(monkeypatch):
    patch = AsyncMock()
    monkeypatch.setattr(
        history_sync, "authorize_import", AsyncMock(side_effect=SlackHistoryError("denied"))
    )
    monkeypatch.setattr(history_sync, "_patch_entry", patch)
    with pytest.raises(SlackHistoryError):
        await history_sync.run_history_import("T1", selection(), user=SimpleNamespace(id=uuid4()))
    patch.assert_not_awaited()


@pytest.mark.asyncio
async def test_job_failure_is_reported_and_retryable(monkeypatch):
    patch = AsyncMock()
    user = SimpleNamespace(id=uuid4())
    request = selection()
    monkeypatch.setattr(history_sync, "authorize_import", AsyncMock())
    monkeypatch.setattr(history_sync, "_patch_entry", patch)
    ingest = AsyncMock(side_effect=RuntimeError("secret must not be persisted"))
    monkeypatch.setattr(history_sync, "import_slack_history", ingest)
    with pytest.raises(RuntimeError):
        await history_sync.run_history_import("T1", request, user=user)
    report = patch.call_args.args[3]
    assert report["status"] == "failed"
    assert "secret" not in str(report)
    assert ("T1", str(request.dataset_id)) not in history_sync._running
    ingest.side_effect = None
    ingest.return_value = SlackHistoryResult(dataset_id=request.dataset_id)
    await history_sync.run_history_import("T1", request, user=user)
    assert patch.call_args.args[3]["status"] == "completed"


@pytest.mark.asyncio
async def test_metadata_patch_refuses_reassigned_connection(monkeypatch):
    monkeypatch.setattr(
        history_sync,
        "get_by_team",
        AsyncMock(
            return_value=SimpleNamespace(
                status="active",
                user_id=uuid4(),
                provider_metadata={},
            )
        ),
    )
    update = AsyncMock()
    monkeypatch.setattr(history_sync, "update_provider_metadata", update)
    await history_sync._patch_entry("T1", "history_reports", "D1", {}, user_id=uuid4())
    update.assert_not_awaited()


def _payload():
    return {
        "type": "view_submission",
        "team": {"id": "T1"},
        "user": {"id": "U1"},
        "view": {
            **handle_history.history_view(
                team_id="T1", response_url="https://hooks.slack.com/commands/fake"
            ),
            "state": {
                "values": {
                    "dataset": {"value": {"selected_option": {"value": str(uuid4())}}},
                    "channels": {"value": {"selected_conversations": ["C1"]}},
                    "days": {"value": {"value": "14"}},
                    "links": {"value": {"value": ""}},
                    "options": {"value": {"selected_options": [{"value": "sync"}]}},
                }
            },
        },
    }


@pytest.mark.asyncio
async def test_modal_validation_error_does_not_start_an_import(monkeypatch):
    monkeypatch.setattr(handle_history, "_owner", AsyncMock())
    payload = _payload()
    payload["view"]["state"]["values"]["dataset"]["value"]["selected_option"]["value"] = (
        "not-a-uuid"
    )
    response = await handle_history.handle_history_interactive(payload)
    assert response["response_action"] == "errors"


@pytest.mark.asyncio
async def test_modal_returns_before_source_fetch_and_uses_authenticated_owner(monkeypatch):
    owner_id = uuid4()
    user = SimpleNamespace(id=owner_id)
    monkeypatch.setattr(
        handle_history, "_owner", AsyncMock(return_value=SimpleNamespace(user_id=owner_id))
    )
    monkeypatch.setattr(handle_history, "get_user", AsyncMock(return_value=user))
    queued = []
    monkeypatch.setattr(handle_history, "background_import", queued.append)
    run = AsyncMock()
    monkeypatch.setattr(handle_history, "_import_and_confirm", run)
    response = await handle_history.handle_history_interactive(_payload())
    assert response == {"response_action": "clear"}
    assert len(queued) == 1
    run.assert_not_awaited()
    await queued[0]
    assert run.call_args.kwargs["user"] is user
    assert run.call_args.kwargs["keep_synced"] is True
    assert run.call_args.args[1].days == 14


@pytest.mark.asyncio
async def test_workspace_tampering_is_rejected(monkeypatch):
    monkeypatch.setattr(handle_history, "_owner", AsyncMock())
    payload = _payload()
    payload["team"]["id"] = "TOTHER"
    response = await handle_history.handle_history_interactive(payload)
    assert response["response_action"] == "errors"


def test_thread_shortcut_keeps_selected_parent_without_importing_whole_channel():
    payload = _payload()
    payload["view"]["private_metadata"] = handle_history.history_view(
        team_id="T1",
        thread={"channel_id": "G1", "ts": "1700000000.000001"},
    )["private_metadata"]
    payload["view"]["state"]["values"]["channels"] = {"value": {"selected_conversations": []}}
    _, chosen, _ = handle_history._submission(payload)
    assert chosen.channel_ids == []
    assert chosen.threads[0].channel_id == "G1"
    assert chosen.threads[0].ts == "1700000000.000001"


@pytest.mark.asyncio
async def test_import_confirmation_only_after_indexing_completes(monkeypatch):
    request = selection()
    started, finish = asyncio.Event(), asyncio.Event()

    async def importing(*args, **kwargs):
        started.set()
        await finish.wait()
        return SlackHistoryResult(dataset_id=request.dataset_id, conversations=1, messages=2)

    monkeypatch.setattr(handle_history, "run_history_import", importing)
    post = AsyncMock()
    monkeypatch.setattr(handle_history, "post_to_response_url", post)
    task = asyncio.create_task(
        handle_history._import_and_confirm(
            {"team_id": "T1", "response_url": "url"},
            request,
            user=SimpleNamespace(id=uuid4()),
            keep_synced=False,
        )
    )
    await started.wait()
    post.assert_not_awaited()
    finish.set()
    await task
    assert post.call_args.args[1]["response_type"] == "ephemeral"
    assert "1 conversations (2 messages)" in post.call_args.args[1]["text"]


def test_history_http_requires_authentication():
    app = FastAPI()
    app.include_router(get_slack_history_router(), prefix="/api/v1/slack")
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/slack/history/T1/import", json=selection().model_dump(mode="json")
        )
    assert response.status_code in (401, 403)


def test_history_http_uses_authenticated_user_and_does_not_accept_user_id(monkeypatch):
    import importlib

    router_module = importlib.import_module("cognee.api.v1.slack.routers.get_slack_history_router")
    user = SimpleNamespace(id=uuid4())
    chosen = selection()
    run = AsyncMock(return_value=SlackHistoryResult(dataset_id=chosen.dataset_id))
    monkeypatch.setattr(router_module, "run_history_import", run)
    app = FastAPI()
    app.include_router(get_slack_history_router(), prefix="/api/v1/slack")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/slack/history/T1/import", json=chosen.model_dump(mode="json")
        )
        forged = client.post(
            "/api/v1/slack/history/T1/import",
            json={
                **chosen.model_dump(mode="json"),
                "user_id": str(uuid4()),
            },
        )
    assert response.status_code == 200
    assert run.call_args.kwargs["user"] is user
    assert forged.status_code == 422


@pytest.mark.asyncio
async def test_dataset_picker_filters_by_all_native_permissions(monkeypatch):
    readable = SimpleNamespace(id=uuid4(), name="Read only")
    full = SimpleNamespace(id=uuid4(), name="Team memory")
    monkeypatch.setattr(
        handle_history, "_owner", AsyncMock(return_value=SimpleNamespace(user_id=uuid4()))
    )
    monkeypatch.setattr(handle_history, "get_user", AsyncMock())
    monkeypatch.setattr(
        handle_history,
        "get_all_user_permission_datasets",
        AsyncMock(
            side_effect=[
                [readable, full],
                [full],
                [full],
            ]
        ),
    )
    payload = _payload()
    payload["type"] = "block_suggestion"
    payload["value"] = "team"
    result = await handle_history.handle_history_interactive(payload)
    assert result["options"] == [
        {"text": {"type": "plain_text", "text": "Team memory"}, "value": str(full.id)}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("hours,enabled,expected", [(5, True, 0), (7, True, 1), (7, False, 0)])
async def test_worker_respects_six_hours_and_disabled_selection(
    monkeypatch, hours, enabled, expected
):
    chosen = selection()
    now = datetime.now(timezone.utc)
    credential = SimpleNamespace(
        id=uuid4(),
        user_id=uuid4(),
        provider_account_id="T1",
        provider_metadata={
            "history_sync": {
                str(chosen.dataset_id): SlackSyncSettings(
                    enabled=enabled, selection=chosen
                ).model_dump(mode="json")
            },
            "history_reports": {
                str(chosen.dataset_id): {
                    "status": "completed",
                    "finished_at": (now - timedelta(hours=hours)).isoformat(),
                }
            },
        },
    )
    result = MagicMock()
    result.scalars.return_value.all.return_value = [credential]
    db = AsyncMock()
    db.execute.return_value = result
    db.__aenter__.return_value = db
    monkeypatch.setattr(
        history_sync, "get_relational_engine", lambda: SimpleNamespace(get_async_session=lambda: db)
    )
    monkeypatch.setattr(history_sync, "get_user", AsyncMock())
    run = AsyncMock()
    monkeypatch.setattr(history_sync, "run_history_import", run)
    await history_sync.sync_due_slack_history()
    assert run.await_count == expected
    if expected:
        assert run.call_args.kwargs["reconcile"] is True
