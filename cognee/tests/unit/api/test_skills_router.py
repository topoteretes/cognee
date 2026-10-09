"""Tests for the dataset-scoped skills router.

Covers DELETE /api/v1/skills/{skill_id} and POST /api/v1/skills error mapping
(#4281): ingest failures must not all collapse to HTTP 409.
"""

from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from cognee.exceptions import CogneeApiError

router_module = import_module("cognee.api.v1.skills.routers.get_skills_router")
list_module = import_module("cognee.api.v1.skills.list_skills")
remember_pkg = import_module("cognee.api.v1.remember")


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router_module.get_skills_router(), prefix="/api/v1/skills")
    return app


def _delete_client(monkeypatch, *, authorized=True, delete_result=True) -> TestClient:
    app = _app()
    user_id = uuid4()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id,
        tenant_id=None,
    )

    async def fake_authorized_datasets(dataset_ids, _permission, _user):
        if not authorized:
            return []
        return [SimpleNamespace(id=dataset_ids[0])]

    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        fake_authorized_datasets,
    )

    async def fake_delete_skill(skill_id, dataset):
        return delete_result

    monkeypatch.setattr(list_module, "delete_skill", fake_delete_skill)
    return TestClient(app)


def test_delete_skill_success(monkeypatch):
    client = _delete_client(monkeypatch)
    skill_id = str(uuid4())
    dataset_id = str(uuid4())

    response = client.delete(
        f"/api/v1/skills/{skill_id}",
        params={"dataset_id": dataset_id},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "deleted",
        "id": skill_id,
        "dataset_id": dataset_id,
    }


def test_delete_skill_forbidden(monkeypatch):
    client = _delete_client(monkeypatch, authorized=False)

    response = client.delete(
        f"/api/v1/skills/{uuid4()}",
        params={"dataset_id": str(uuid4())},
    )

    assert response.status_code == 403
    assert response.json() == {"error": "Not authorized for this dataset"}


def test_delete_skill_not_found(monkeypatch):
    client = _delete_client(monkeypatch, delete_result=False)

    response = client.delete(
        f"/api/v1/skills/{uuid4()}",
        params={"dataset_id": str(uuid4())},
    )

    assert response.status_code == 404
    assert response.json() == {"error": "Skill not found"}


def test_delete_skill_requires_dataset_query_param(monkeypatch):
    client = _delete_client(monkeypatch)

    response = client.delete(f"/api/v1/skills/{uuid4()}")

    assert response.status_code == 422


def _ingest_client(monkeypatch, *, remember_side_effect=None, remember_result=None) -> TestClient:
    app = _app()

    @app.exception_handler(CogneeApiError)
    async def _cognee_api_error(_: Request, exc: CogneeApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": f"{exc.message} [{exc.name}]"},
        )

    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(),
        tenant_id=None,
    )
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)

    mock_remember = AsyncMock()
    if remember_side_effect is not None:
        mock_remember.side_effect = remember_side_effect
    else:
        mock_remember.return_value = remember_result or SimpleNamespace(
            to_dict=lambda: {"status": "completed"}
        )
    monkeypatch.setattr(remember_pkg, "remember", mock_remember)
    return TestClient(app)


def _ingest_payload(**overrides):
    body = {
        "skills_text": "---\nname: demo\n---\nDo the thing.",
        "dataset_name": "skills_dataset",
    }
    body.update(overrides)
    return body


def test_ingest_skill_success(monkeypatch):
    client = _ingest_client(monkeypatch)

    response = client.post("/api/v1/skills", json=_ingest_payload())

    assert response.status_code == 200
    assert response.json() == {"status": "completed"}


def test_ingest_skill_requires_dataset(monkeypatch):
    client = _ingest_client(monkeypatch)

    response = client.post(
        "/api/v1/skills",
        json={"skills_text": "Do the thing."},
    )

    assert response.status_code == 400
    assert response.json() == {"error": "Either dataset_name or dataset_id is required"}


def test_ingest_skill_malformed_input_returns_400(monkeypatch):
    client = _ingest_client(
        monkeypatch,
        remember_side_effect=ValueError("Invalid skill filename: ../escape.md"),
    )

    response = client.post("/api/v1/skills", json=_ingest_payload())

    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "Failed to ingest skill"
    assert body["exception_type"] == "ValueError"
    assert "Invalid skill filename" in body["detail"]


def test_ingest_skill_internal_error_returns_500(monkeypatch):
    client = _ingest_client(
        monkeypatch,
        remember_side_effect=RuntimeError("Graph edge indexing error"),
    )

    response = client.post("/api/v1/skills", json=_ingest_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "Failed to ingest skill"
    assert body["exception_type"] == "RuntimeError"
    assert body["detail"] == "Graph edge indexing error"


def test_ingest_skill_parse_keyerror_returns_500_with_type(monkeypatch):
    client = _ingest_client(monkeypatch, remember_side_effect=KeyError("name"))

    response = client.post("/api/v1/skills", json=_ingest_payload())

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "Failed to ingest skill"
    assert body["exception_type"] == "KeyError"
    assert "name" in body["detail"]


def test_ingest_skill_conflict_preserves_409(monkeypatch):
    client = _ingest_client(
        monkeypatch,
        remember_side_effect=CogneeApiError(
            message="Skill already exists",
            name="SkillConflictError",
            status_code=409,
            log=False,
        ),
    )

    response = client.post("/api/v1/skills", json=_ingest_payload())

    assert response.status_code == 409
    assert "Skill already exists" in response.json()["detail"]
