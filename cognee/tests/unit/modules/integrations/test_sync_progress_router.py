"""The progress route shares the integrations connection ownership boundary."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import cognee.api.v1.integrations.routers.get_sync_progress_router as route_module
from cognee.modules.integrations.credentials import CredentialInactiveError
from cognee.modules.integrations.sync_progress import SyncProgress, SyncProgressResult
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture
def route(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    credential = SimpleNamespace(id=uuid4(), user_id=user.id)
    find = AsyncMock(return_value=credential)
    read = AsyncMock(return_value=SyncProgressResult())
    monkeypatch.setattr(route_module, "get_integration", lambda provider: object())
    monkeypatch.setattr(route_module, "get_active_credential_for_user", find)
    monkeypatch.setattr(route_module, "get_sync_progress", read)
    app = FastAPI()
    app.include_router(route_module.get_sync_progress_router(), prefix="/integrations")
    app.dependency_overrides[get_authenticated_user] = lambda: user
    return TestClient(app), find, read, credential, app


def test_account_filter_and_empty_response(route):
    client, find, read, credential, _ = route
    response = client.get("/integrations/github/sync/progress?account_id=account-2")
    assert response.status_code == 200
    assert response.json() == {"latest": None, "activeRuns": [], "unknownActivityCount": 0}
    find.assert_awaited_once_with(credential.user_id, "github", "account-2")
    read.assert_awaited_once_with(credential)


def test_missing_account_does_not_fall_back_to_another_connection(route):
    client, find, read, _, _ = route
    find.return_value = None
    assert client.get("/integrations/github/sync/progress?account_id=not-mine").status_code == 404
    read.assert_not_awaited()


def test_unknown_provider_before_connection_lookup(route, monkeypatch):
    client, find, _, _, _ = route

    def unknown(provider):
        raise KeyError(provider)

    monkeypatch.setattr(route_module, "get_integration", unknown)
    assert client.get("/integrations/unknown/sync/progress").status_code == 404
    find.assert_not_awaited()


def test_revocation_between_route_auth_and_storage_read(route):
    client, _, read, _, _ = route
    read.side_effect = CredentialInactiveError("private details")
    response = client.get("/integrations/github/sync/progress")
    assert response.status_code == 404
    assert "private details" not in response.text


def test_auth_required_before_storage_read(route):
    client, find, _, _, app = route

    def unauthorized():
        raise HTTPException(status_code=401)

    app.dependency_overrides[get_authenticated_user] = unauthorized
    assert client.get("/integrations/github/sync/progress").status_code == 401
    find.assert_not_awaited()


def test_nested_progress_uses_camel_case_and_unknown_liveness(route):
    client, _, read, credential, _ = route
    now = datetime.now(timezone.utc)
    read.return_value = SyncProgressResult(
        latest=SyncProgress(
            run_id=uuid4(),
            provider="github",
            credential_id=credential.id,
            started_at=now,
            phase_started_at=now,
            updated_at=now,
        ),
        unknown_activity_count=1,
    )
    response = client.get("/integrations/github/sync/progress")
    assert response.status_code == 200
    payload = response.json()
    assert payload["latest"]["executionActive"] is None
    assert payload["latest"]["phaseEtaSeconds"] is None
    assert "startedAt" in payload["latest"]
    assert "started_at" not in payload["latest"]
    assert payload["unknownActivityCount"] == 1
