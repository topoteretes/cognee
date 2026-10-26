"""Scoped cognify validation uses HTTP 422, not the ontology-conflict handler."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.client import exception_handler
from cognee.api.v1.cognify.routers.get_cognify_router import get_cognify_router
from cognee.exceptions import CogneeApiError
from cognee.modules.users.methods import get_authenticated_user


@pytest.mark.parametrize("datasets, ids", [(["a", "b"], [uuid4()]), (["a"], [])])
def test_scoped_selection_errors_are_422(monkeypatch, datasets, ids):
    migrations = importlib.import_module("cognee.modules.migrations.startup")
    methods = importlib.import_module("cognee.modules.data.methods")
    cognify = importlib.import_module("cognee.api.v1.cognify.cognify")
    monkeypatch.setattr(migrations, "run_migrations_and_block", AsyncMock())
    monkeypatch.setattr(
        methods,
        "get_authorized_existing_datasets",
        AsyncMock(return_value=[SimpleNamespace(id=uuid4())]),
    )
    monkeypatch.setattr(methods, "get_dataset_data", AsyncMock(return_value=[]))
    executor = AsyncMock()
    monkeypatch.setattr(cognify, "get_pipeline_executor", executor)
    app = FastAPI()
    app.add_exception_handler(CogneeApiError, exception_handler)
    app.include_router(get_cognify_router(), prefix="/api/v1/cognify")
    app.dependency_overrides[get_authenticated_user] = lambda: SimpleNamespace(id=uuid4())
    response = TestClient(app).post(
        "/api/v1/cognify", json={"datasets": datasets, "data_ids": [str(i) for i in ids]}
    )
    assert response.status_code == 422, response.text
    executor.assert_not_called()
