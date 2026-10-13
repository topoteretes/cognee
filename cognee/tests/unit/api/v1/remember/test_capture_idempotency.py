"""The typed HTTP endpoint exposes stable capture IDs and explicit conflicts."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.remember.routers.get_remember_router import get_remember_router
from cognee.infrastructure.databases.cache.sql.capture import CaptureConflictError
from cognee.modules.users.methods import get_authenticated_user


@pytest.fixture
def capture(monkeypatch):
    stub = AsyncMock(return_value={"entry_id": "entry", "client_event_id": "event"})
    monkeypatch.setattr("cognee.infrastructure.session.capture.remember_capture", stub)
    return stub


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(get_remember_router(), prefix="/remember")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    return TestClient(app)


def body():
    return {
        "entry": {"type": "qa", "question": "q", "answer": "a"},
        "session_id": "sid",
        "dataset_id": str(uuid4()),
        "client_event_id": str(uuid4()),
    }


def test_capture_id_and_dataset_uuid_are_forwarded(client, capture):
    payload = body()
    response = client.post("/remember/entry", json=payload)
    assert response.status_code == 200
    args = capture.call_args.kwargs
    assert str(args["client_event_id"]) == payload["client_event_id"]
    assert str(args["dataset_id"]) == payload["dataset_id"]
    assert args["entry"].answer == "a" and args["session_id"] == "sid"


def test_conflicting_payload_is_distinguishable_from_transient_errors(client, capture):
    capture.side_effect = CaptureConflictError("do not echo the original data")
    response = client.post("/remember/entry", json=body())
    assert response.status_code == 409
    assert response.json()["code"] == "capture_payload_conflict"


def test_invalid_event_id_is_rejected_before_writing(client, capture):
    response = client.post("/remember/entry", json={**body(), "client_event_id": "not-a-uuid"})
    assert response.status_code == 422
    capture.assert_not_awaited()


def test_unsupported_cache_fails_without_falling_back_to_a_duplicate_write(client, capture):
    capture.side_effect = RuntimeError("unsupported cache")
    assert client.post("/remember/entry", json=body()).status_code == 503


def test_capture_does_not_run_skill_improvement(client, capture):
    assert (
        client.post("/remember/entry", json={**body(), "skill_improvement": {}}).status_code == 400
    )
    capture.assert_not_awaited()
