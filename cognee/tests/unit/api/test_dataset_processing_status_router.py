"""Router test for GET /v1/datasets/{dataset_id}/processing-status (SDK-23).

Covers what a router-level test actually exercises: the real auth dependency
is enforced, the dataset permission check gates the response, the ``pipeline``
query param reaches the helper, and the ``{total, completed, pending}`` body
reaches the client unchanged. Counting semantics live in
tests/unit/modules/data/test_get_dataset_processing_status.py.
"""

import importlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user

ROUTER_MODULE = "cognee.api.v1.datasets.routers.get_datasets_router"


@pytest.fixture(scope="module")
def test_client():
    from cognee.api.v1.datasets.routers.get_datasets_router import get_datasets_router

    app = FastAPI()
    app.include_router(get_datasets_router(), prefix="/api/v1/datasets")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def authenticated_client(test_client, monkeypatch):
    router_module = importlib.import_module(ROUTER_MODULE)
    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)

    async def _override_user():
        return SimpleNamespace(
            id=str(uuid.uuid4()),
            email="default@example.com",
            is_active=True,
            tenant_id=str(uuid.uuid4()),
        )

    test_client.app.dependency_overrides[get_authenticated_user] = _override_user
    yield test_client
    test_client.app.dependency_overrides.pop(get_authenticated_user, None)


def _authorize_datasets(monkeypatch, datasets):
    router_module = importlib.import_module(ROUTER_MODULE)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=datasets),
    )


def _patch_status_helper(monkeypatch, helper):
    # The route imports the helper from the package inside the handler, so the
    # package attribute is what it resolves at call time.
    methods_module = importlib.import_module("cognee.modules.data.methods")
    monkeypatch.setattr(methods_module, "get_dataset_processing_status", helper)


def test_processing_status_requires_authentication(test_client):
    """No dependency override, no auth cookie/header — the real
    get_authenticated_user dependency must reject the request rather than
    the route ever running."""
    response = test_client.get(f"/api/v1/datasets/{uuid.uuid4()}/processing-status")

    assert response.status_code in (401, 403)


def test_processing_status_returns_counts(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_datasets(monkeypatch, [SimpleNamespace(id=dataset_id)])

    helper = AsyncMock(return_value={"total": 10, "completed": 7, "pending": 3})
    _patch_status_helper(monkeypatch, helper)

    response = authenticated_client.get(f"/api/v1/datasets/{dataset_id}/processing-status")

    assert response.status_code == 200
    # Exactly the three counts: the optional items field must not leak as null.
    assert response.json() == {"total": 10, "completed": 7, "pending": 3}
    helper.assert_awaited_once_with(
        dataset_id, pipeline_name="cognify_pipeline", include_items=False
    )


def test_processing_status_include_items_returns_breakdown(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    done_id, pending_id = uuid.uuid4(), uuid.uuid4()
    _authorize_datasets(monkeypatch, [SimpleNamespace(id=dataset_id)])

    helper = AsyncMock(
        return_value={
            "total": 2,
            "completed": 1,
            "pending": 1,
            "items": [
                {"id": done_id, "name": "done.pdf", "completed": True},
                {"id": pending_id, "name": "fresh.md", "completed": False},
            ],
        }
    )
    _patch_status_helper(monkeypatch, helper)

    response = authenticated_client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"include_items": "true"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "total": 2,
        "completed": 1,
        "pending": 1,
        "items": [
            {"id": str(done_id), "name": "done.pdf", "completed": True},
            {"id": str(pending_id), "name": "fresh.md", "completed": False},
        ],
    }
    helper.assert_awaited_once_with(
        dataset_id, pipeline_name="cognify_pipeline", include_items=True
    )


def test_processing_status_empty_dataset(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_datasets(monkeypatch, [SimpleNamespace(id=dataset_id)])
    _patch_status_helper(
        monkeypatch, AsyncMock(return_value={"total": 0, "completed": 0, "pending": 0})
    )

    response = authenticated_client.get(f"/api/v1/datasets/{dataset_id}/processing-status")

    assert response.status_code == 200
    assert response.json() == {"total": 0, "completed": 0, "pending": 0}


def test_processing_status_forwards_pipeline_query_param(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_datasets(monkeypatch, [SimpleNamespace(id=dataset_id)])

    helper = AsyncMock(return_value={"total": 2, "completed": 2, "pending": 0})
    _patch_status_helper(monkeypatch, helper)

    response = authenticated_client.get(
        f"/api/v1/datasets/{dataset_id}/processing-status",
        params={"pipeline": "add_pipeline"},
    )

    assert response.status_code == 200
    helper.assert_awaited_once_with(dataset_id, pipeline_name="add_pipeline", include_items=False)


def test_processing_status_unknown_or_unauthorized_dataset_is_404(
    authenticated_client, monkeypatch
):
    dataset_id = uuid.uuid4()
    _authorize_datasets(monkeypatch, [])

    helper = AsyncMock()
    _patch_status_helper(monkeypatch, helper)

    response = authenticated_client.get(f"/api/v1/datasets/{dataset_id}/processing-status")

    assert response.status_code == 404
    assert response.json() == {"message": f"Dataset ({dataset_id}) not found."}
    helper.assert_not_awaited()


def test_processing_status_error_returns_409(authenticated_client, monkeypatch):
    dataset_id = uuid.uuid4()
    _authorize_datasets(monkeypatch, [SimpleNamespace(id=dataset_id)])
    _patch_status_helper(monkeypatch, AsyncMock(side_effect=RuntimeError("db unavailable")))

    response = authenticated_client.get(f"/api/v1/datasets/{dataset_id}/processing-status")

    assert response.status_code == 409
    assert response.json() == {"error": "Unable to retrieve dataset processing status."}


def test_processing_status_invalid_dataset_id_is_422(authenticated_client):
    response = authenticated_client.get("/api/v1/datasets/not-a-uuid/processing-status")

    assert response.status_code == 422
