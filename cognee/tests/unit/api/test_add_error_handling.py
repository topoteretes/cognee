import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.api.v1.add.routers.get_add_router import get_add_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_add_router(), prefix="/api/v1/add")

    mock_user = SimpleNamespace(
        id=uuid4(), email="default@example.com", is_active=True, tenant_id=uuid4()
    )

    async def override_user():
        return mock_user

    app.dependency_overrides[get_authenticated_user] = override_user
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_add_pipeline_run_errored_returns_422_error_response(client):
    # Patch the function imported by the router: from cognee.api.v1.add import add as cognee_add
    import cognee.api.v1.add as add_pkg

    from uuid import uuid4

    add_pkg.add = AsyncMock(
        return_value=PipelineRunErrored(
            pipeline_run_id=uuid4(),
            dataset_id=uuid4(),
            dataset_name="test_dataset",
            status="errored",
            error="pipeline failed",
        )
    )

    # multipart form data
    resp = client.post(
        "/api/v1/add",
        data={"datasetName": "test_dataset"},
        files={"data": ("x.txt", b"hello", "text/plain")},
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "Pipeline run errored"
    assert "detail" in body