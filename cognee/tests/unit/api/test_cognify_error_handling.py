import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_cognify_router(), prefix="/api/v1/cognify")

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


def test_cognify_pipeline_run_errored_returns_422_error_response(client):
    # Router does: from cognee.api.v1.cognify import cognify as cognee_cognify
    import cognee.api.v1.cognify as cognify_pkg

    errored = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="test_dataset",
        status="errored",
        error="pipeline failed",
    )

    cognify_pkg.cognify = AsyncMock(return_value={"run": errored})

    resp = client.post(
        "/api/v1/cognify",
        json={
            "datasets": ["test_dataset"],
            "run_in_background": False,
        },
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "Pipeline run errored"
    assert "detail" in body