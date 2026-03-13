import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.exceptions.exceptions import PermissionDeniedError
from cognee.modules.users.methods import get_authenticated_user
from cognee.api.v1.search.routers.get_search_router import get_search_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_search_router(), prefix="/api/v1/search")

    # Override auth dependency
    mock_user = SimpleNamespace(
        id=uuid4(),
        email="default@example.com",
        is_active=True,
        tenant_id=uuid4(),
    )

    async def override_user():
        return mock_user

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_search_permission_denied_returns_403_error_response(app, client, monkeypatch):
    # Patch the actual implementation used by the router at runtime:
    # router does: from cognee.api.v1.search import search as cognee_search
    # and __init__.py re-exports search from .search
    import cognee.api.v1.search as search_pkg

    search_pkg.search = AsyncMock(side_effect=PermissionDeniedError("no access to dataset"))

    resp = client.post(
        "/api/v1/search",
        json={
            "search_type": "GRAPH_COMPLETION",
            "datasets": ["some_dataset"],
            "dataset_ids": [],
            "query": "What is in the document?",
            "system_prompt": "Answer briefly.",
            "node_name": [],
            "top_k": 10,
            "only_context": False,
            "verbose": False,
        },
    )

    assert resp.status_code == 403
    body = resp.json()
    assert body["error"] == "Permission denied"
    assert "detail" in body
    assert "hint" not in body