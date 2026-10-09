"""Tests for the dataset-scoped skills router (DELETE /api/v1/skills/{skill_id})."""

from importlib import import_module
from types import SimpleNamespace
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


def _list_client(monkeypatch, *, authorized=True, skills=None, total=0) -> TestClient:
    """Client for the list/count routes, recording the arguments they forward."""
    app = _app()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(),
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

    calls: dict = {}

    async def fake_list_skills(**kwargs):
        calls["list"] = kwargs
        return skills or []

    async def fake_count_skills(**kwargs):
        calls["count"] = kwargs
        return total

    monkeypatch.setattr(list_module, "list_skills", fake_list_skills)
    monkeypatch.setattr(list_module, "count_skills", fake_count_skills)

    client = TestClient(app)
    client.calls = calls
    return client


def test_list_skills_forwards_pagination(monkeypatch):
    client = _list_client(monkeypatch)

    response = client.get(
        "/api/v1/skills/",
        params={"dataset_id": str(uuid4()), "limit": 1000, "offset": 200},
    )

    assert response.status_code == 200
    assert client.calls["list"]["limit"] == 1000
    assert client.calls["list"]["offset"] == 200


def test_count_skills_returns_total(monkeypatch):
    """The count is the whole dataset, not the page the list route would return."""
    client = _list_client(monkeypatch, skills=[{"id": "a"}], total=378)
    dataset_id = str(uuid4())

    response = client.get("/api/v1/skills/count", params={"dataset_id": dataset_id})

    assert response.status_code == 200
    assert response.json() == {"count": 378}
    # Proves /count reached the count handler rather than being captured as a
    # skill_id by GET /{skill_id}, which is declared after it.
    assert client.calls["count"]["include_inactive"] is False


def test_count_skills_honors_include_inactive(monkeypatch):
    client = _list_client(monkeypatch, total=5)

    response = client.get(
        "/api/v1/skills/count",
        params={"dataset_id": str(uuid4()), "include_inactive": "true"},
    )

    assert response.status_code == 200
    assert client.calls["count"]["include_inactive"] is True


def test_count_skills_forbidden(monkeypatch):
    client = _list_client(monkeypatch, authorized=False)

    response = client.get("/api/v1/skills/count", params={"dataset_id": str(uuid4())})

    assert response.status_code == 403
    assert response.json() == {"error": "Not authorized for this dataset"}
