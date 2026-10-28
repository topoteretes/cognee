"""Settings, preview and due-run endpoints enforce account ownership."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import cognee.api.v1.integrations.routers.get_github_sync_router as routes
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture
def api(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    credential = SimpleNamespace(
        id=uuid4(), user_id=user.id, provider_metadata={}, last_synced_at=None, sync_status=None
    )
    find = AsyncMock(return_value=credential)
    save = AsyncMock(return_value=credential)
    preview = AsyncMock(
        return_value={"repository_id": "1", "branch": "main", "selected": [], "skipped": {}}
    )
    due = AsyncMock(return_value={})
    spawned = []
    monkeypatch.setattr(routes, "get_active_credential_for_user", find)
    monkeypatch.setattr(routes, "save_sync_settings", save)
    monkeypatch.setattr(routes, "preview_repository", preview)
    monkeypatch.setattr(routes, "run_due_github_syncs", due)
    monkeypatch.setattr(
        routes, "spawn_background", lambda coro, **kwargs: (spawned.append(kwargs), coro.close())
    )
    app = FastAPI()
    app.include_router(routes.get_github_sync_router(), prefix="/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return SimpleNamespace(
        client=TestClient(app),
        user=user,
        credential=credential,
        find=find,
        save=save,
        preview=preview,
        due=due,
        spawned=spawned,
        app=app,
    )


def test_get_settings_uses_owned_account(api):
    response = api.client.get("/integrations/github/sync/settings?account_id=42")
    assert response.status_code == 200
    assert response.json()["settings"]["enabled"] is True
    assert response.json()["settings"]["mode"] == "push"
    api.find.assert_awaited_once_with(api.user.id, "github", "42")


def test_save_rejects_unknown_fields_and_invalid_schedule(api):
    response = api.client.put(
        "/integrations/github/sync/settings",
        json={"settings": {"mode": "scheduled", "schedule": {"interval_minutes": 0}}},
    )
    assert response.status_code == 422
    api.save.assert_not_awaited()


def test_pause_is_saved_without_launching_sync(api):
    response = api.client.put(
        "/integrations/github/sync/settings", json={"settings": {"enabled": False}}
    )
    assert response.status_code == 200
    assert response.json()["syncRequested"] is False
    assert api.save.await_args.args[1].enabled is False


def test_first_scheduled_selection_waits_for_its_slot(api):
    response = api.client.put(
        "/integrations/github/sync/settings",
        json={
            "settings": {
                "mode": "scheduled",
                "first_sync": "scheduled",
                "schedule": {"interval_minutes": 15},
                "repositories": {"1": {}},
            },
            "acknowledge_removal": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["syncRequested"] is False


def test_explicit_sync_now_while_paused_uses_manual_trigger(api, monkeypatch):
    sync = AsyncMock()
    monkeypatch.setattr(routes, "sync_repositories", sync)
    response = api.client.put(
        "/integrations/github/sync/settings",
        json={"settings": {"enabled": False}, "sync_now": True},
    )
    assert response.status_code == 200
    assert response.json()["syncRequested"] is True
    sync.assert_called_once_with(api.credential, trigger="manual")


def test_save_passes_explicit_removal_acknowledgement(api):
    response = api.client.put(
        "/integrations/github/sync/settings",
        json={"settings": {"enabled": False, "repositories": {}}, "acknowledge_removal": True},
    )
    assert response.status_code == 200
    assert api.save.await_args.kwargs["acknowledge_removal"] is True


@pytest.mark.parametrize("path", ["settings", "preview"])
def test_other_users_account_is_not_accessible(api, path):
    api.find.return_value = None
    response = (
        api.client.get("/integrations/github/sync/settings?account_id=other")
        if path == "settings"
        else api.client.post(
            "/integrations/github/sync/preview?account_id=other",
            json={"repository_id": "1", "settings": {}},
        )
    )
    assert response.status_code == 404
    api.preview.assert_not_awaited()


def test_revocation_race_does_not_expose_storage_error(api):
    api.save.side_effect = CredentialInactiveError("private")
    response = api.client.put("/integrations/github/sync/settings", json={"settings": {}})
    assert response.status_code == 404
    assert "private" not in response.text


def test_due_runner_is_user_scoped_and_runs_in_the_background(api):
    response = api.client.post("/integrations/github/sync/due")
    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    api.due.assert_called_once_with(user_id=api.user.id)
    assert api.spawned == [{"description": "Due GitHub syncs"}]


def test_due_runner_starts_nothing_without_a_connection(api):
    api.find.return_value = None
    response = api.client.post("/integrations/github/sync/due")
    assert response.json() == {"accepted": False}
    api.due.assert_not_called()
    assert api.spawned == []


def test_requests_accept_camel_case_and_responses_use_it(api):
    # The save is mocked to return this connection, so its stored policy is what the reply shows.
    api.credential.provider_metadata = {
        "github_sync_settings": {"mode": "scheduled", "schedule": {"interval_minutes": 15}}
    }
    response = api.client.put(
        "/integrations/github/sync/settings",
        json={
            "settings": {
                "mode": "scheduled",
                "firstSync": "scheduled",
                "schedule": {"intervalMinutes": 15, "startTime": "09:00", "endTime": "17:00"},
                "repositories": {"1": {"includePaths": ["src/**"]}},
            },
            "acknowledgeRemoval": True,
            "syncNow": False,
        },
    )
    assert response.status_code == 200
    saved = api.save.await_args.args[1]
    assert saved.schedule.interval_minutes == 15
    assert saved.first_sync == "scheduled"
    assert saved.repositories["1"].include_paths == ["src/**"]
    assert api.save.await_args.kwargs["acknowledge_removal"] is True
    body = response.json()
    assert body["settings"]["schedule"]["intervalMinutes"] == 15
    assert "firstSync" in body["settings"]
    assert {"nextSyncAt", "lastSyncedAt", "syncStatus", "syncRequested"} <= set(body)


def test_preview_accepts_camel_case_and_answers_with_it(api):
    response = api.client.post(
        "/integrations/github/sync/preview",
        json={"repositoryId": "1", "settings": {"excludePaths": ["docs/**"]}},
    )
    assert response.status_code == 200
    assert response.json() == {
        "repositoryId": "1",
        "branch": "main",
        "treeSha": None,
        "selected": [],
        "skipped": {},
    }


def test_auth_required(api):
    def unauthorized():
        raise HTTPException(status_code=401)

    api.app.dependency_overrides[get_authenticated_user] = unauthorized
    assert api.client.get("/integrations/github/sync/settings").status_code == 401
    api.find.assert_not_awaited()
