"""POST /api/v1/improve passes the SDK-593 status answers through as HTTP 200 JSON.

The router's response model describes a pipeline-run mapping; ``busy`` /
``no_op`` / ``accepted`` are plain status objects and must reach the client
unchanged (a plugin treats any 2xx dict as a landed submit).
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

router_module = importlib.import_module("cognee.api.v1.improve.routers.get_improve_router")
improve_pkg = importlib.import_module("cognee.api.v1.improve")


def _client(monkeypatch, improve_result):
    app = FastAPI()
    app.include_router(router_module.get_improve_router(), prefix="/api/v1/improve")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    monkeypatch.setattr(router_module, "send_telemetry", lambda *a, **k: None)

    captured = {}

    async def fake_improve(**kwargs):
        captured.update(kwargs)
        return improve_result

    monkeypatch.setattr(improve_pkg, "improve", fake_improve)
    return TestClient(app), captured


@pytest.mark.parametrize(
    "body",
    [
        {"status": "no_op", "dataset_id": "d", "session_ids": ["s1"], "reason": "nothing_pending"},
        {
            "status": "busy",
            "dataset_id": "d",
            "session_ids": ["s1"],
            "session_id": "s1",
            "holder_age_seconds": 3.0,
            "rerun_requested": True,
        },
        {
            "status": "accepted",
            "dataset_id": "d",
            "session_ids": ["s1"],
            "background": True,
            "pending_stages": ["persist_sessions"],
        },
    ],
)
def test_status_bodies_are_returned_verbatim_with_200(monkeypatch, body):
    client, captured = _client(monkeypatch, body)

    response = client.post(
        "/api/v1/improve",
        json={"datasetName": "ds", "sessionIds": ["s1"], "runInBackground": True},
    )

    assert response.status_code == 200
    assert response.json() == body
    assert captured["session_ids"] == ["s1"]
    assert captured["run_in_background"] is True


def test_missing_dataset_is_400(monkeypatch):
    client, _ = _client(monkeypatch, {})

    response = client.post("/api/v1/improve", json={"sessionIds": ["s1"]})

    assert response.status_code == 400
