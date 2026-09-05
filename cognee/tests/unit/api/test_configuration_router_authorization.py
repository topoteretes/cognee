from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient


router_module = import_module("cognee.api.v1.users.routers.get_configuration_router")


def test_get_configuration_scopes_lookup_to_authenticated_user(monkeypatch):
    principal_id = uuid4()
    config_id = uuid4()
    get_configuration = AsyncMock(return_value={})
    monkeypatch.setattr(
        router_module,
        "method_get_principal_configuration",
        get_configuration,
    )

    app = FastAPI()
    app.include_router(router_module.get_configuration_router(), prefix="/configuration")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=principal_id
    )

    response = TestClient(app).get(f"/configuration/get_user_configuration/{config_id}")

    assert response.status_code == 200
    get_configuration.assert_awaited_once_with(
        config_id=config_id,
        principal_id=principal_id,
    )
