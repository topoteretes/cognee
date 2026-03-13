import pytest
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunErrored
from cognee.api.v1.update.routers.get_update_router import get_update_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(get_update_router(), prefix="/api/v1/update")

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


def test_update_pipeline_run_errored_returns_422_error_response(client):
    # Router does: from cognee.api.v1.update import update as cognee_update
    import cognee.api.v1.update as update_pkg

    errored = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="test_dataset",
        status="errored",
        error="update failed",
    )

    update_pkg.update = AsyncMock(return_value={"run": errored})

    resp = client.patch(
        "/api/v1/update",
        params={"data_id": str(uuid4()), "dataset_id": str(uuid4())},
        files={"data": ("x.txt", b"hello", "text/plain")},
        data={"node_set": ""},  # form field
    )

    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "Pipeline run errored"
    assert "detail" in body