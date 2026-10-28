"""Disk exhaustion keeps its HTTP meaning after pipeline error serialization."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.client import exception_handler
from cognee.api.v1.add.routers.get_add_router import get_add_router
from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router
from cognee.exceptions import CogneeApiError
from cognee.modules.pipelines.models import PipelineRunErrored
from cognee.modules.users.methods import get_authenticated_user


@pytest.mark.parametrize("operation", ["add", "cognify"])
@pytest.mark.parametrize(
    "error_class, expected", [("RepositoryResourceError", 507), ("TypeError", 500)]
)
def test_serialized_resource_failure_is_507(monkeypatch, operation, error_class, expected):
    failure = PipelineRunErrored(
        pipeline_run_id=uuid4(),
        dataset_id=uuid4(),
        dataset_name="code",
        error_class=error_class,
        # Deliberately do not rely on matching a class name in legacy repr text.
        payload="Configured repository disk budget exceeded; increase available storage.",
    )
    package = importlib.import_module(f"cognee.api.v1.{operation}")
    run = AsyncMock(return_value=failure if operation == "add" else {failure.dataset_id: failure})
    monkeypatch.setattr(package, operation, run)
    router_module = importlib.import_module(
        f"cognee.api.v1.{operation}.routers.get_{operation}_router"
    )
    monkeypatch.setattr(router_module, "send_telemetry", lambda *a, **kw: None)
    app = FastAPI()
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.include_router(
        get_add_router() if operation == "add" else get_cognify_router(), prefix="/run"
    )
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)
    response = (
        client.post("/run", data={"raw_data": "https://github.com/org/repo", "datasetName": "code"})
        if operation == "add"
        else client.post("/run", json={"datasets": ["code"]})
    )
    assert response.status_code == expected, response.text
    assert "disk budget exceeded" in response.json()["detail"]
    run.assert_awaited_once()


@pytest.mark.parametrize("operation", ["add", "cognify"])
def test_direct_resource_failure_uses_api_exception_handler(monkeypatch, operation):
    from cognee.tasks.code_graph.repo_resources import RepositoryResourceError

    package = importlib.import_module(f"cognee.api.v1.{operation}")
    monkeypatch.setattr(
        package,
        operation,
        AsyncMock(
            side_effect=RepositoryResourceError("disk budget exceeded", 1024, used_bytes=2048)
        ),
    )
    router_module = importlib.import_module(
        f"cognee.api.v1.{operation}.routers.get_{operation}_router"
    )
    monkeypatch.setattr(router_module, "send_telemetry", lambda *a, **kw: None)
    app = FastAPI()
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.include_router(
        get_add_router() if operation == "add" else get_cognify_router(), prefix="/run"
    )
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    client = TestClient(app)
    response = (
        client.post("/run", data={"raw_data": "https://github.com/org/repo", "datasetName": "code"})
        if operation == "add"
        else client.post("/run", json={"datasets": ["code"]})
    )
    assert response.status_code == 507, response.text
    assert "2048 bytes used; budget 1024 bytes" in response.json()["detail"]
