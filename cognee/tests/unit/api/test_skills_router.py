"""Tests for the dataset-scoped skills router (DELETE /api/v1/skills/{skill_id})."""

from types import SimpleNamespace
from importlib import import_module
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

router_module = import_module("cognee.api.v1.skills.routers.get_skills_router")
list_module = import_module("cognee.api.v1.skills.list_skills")


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
