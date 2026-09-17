"""Regression test: cognify/memify error responses surface the real error.

When a cognify/memify pipeline run errors, the routers build the 500 response
``detail`` from ``getattr(run, "error", None) or str(run)``. ``PipelineRunErrored``
has no ``error`` attribute — the failing task's message is carried on ``payload``
(set to ``repr(error)`` by the pipeline runner; the add router reads it
correctly). So ``getattr(..., "error", None)`` was always ``None`` and the detail
fell back to the model's ``str()`` repr instead of the actionable message. The
routers now read ``payload`` like the add router does.
"""

import importlib
import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.modules.users.methods import get_authenticated_user

cognify_router_module = importlib.import_module("cognee.api.v1.cognify.routers.get_cognify_router")
memify_router_module = importlib.import_module("cognee.api.v1.memify.routers.get_memify_router")


def _errored(payload_message):
    return PipelineRunErrored(
        pipeline_run_id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        dataset_name="ds",
        payload=payload_message,
    )


@pytest.fixture(scope="module")
def test_client():
    app = FastAPI()
    app.include_router(cognify_router_module.get_cognify_router(), prefix="/api/v1/cognify")
    app.include_router(memify_router_module.get_memify_router(), prefix="/api/v1/memify")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client(test_client):
    async def override_get_authenticated_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()), email="default@example.com", is_active=True, tenant_id=None
        )

    cognify_router_module.send_telemetry = lambda *a, **k: None
    memify_router_module.send_telemetry = lambda *a, **k: None
    test_client.app.dependency_overrides[get_authenticated_user] = override_get_authenticated_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def test_cognify_error_detail_is_the_task_message(client, monkeypatch):
    message = "LLM_API_KEY is missing"

    import cognee.api.v1.cognify as cognify_pkg

    async def fake_cognify(*_args, **_kwargs):
        return {"ds": _errored(message)}

    monkeypatch.setattr(cognify_pkg, "cognify", fake_cognify)

    response = client.post("/api/v1/cognify", json={"datasets": ["ds"]})

    assert response.status_code == 500
    # The clean task message, not the PipelineRunErrored repr.
    assert response.json()["detail"] == message


def test_memify_error_detail_is_the_task_message(client, monkeypatch):
    message = "enrichment task blew up"

    import cognee.modules.memify as memify_pkg

    async def fake_memify(*_args, **_kwargs):
        return _errored(message)

    monkeypatch.setattr(memify_pkg, "memify", fake_memify)

    response = client.post("/api/v1/memify", json={"dataset_name": "ds"})

    assert response.status_code == 500
    assert response.json()["detail"] == message
