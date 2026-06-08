from types import SimpleNamespace
from importlib import import_module
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

inventory_module = import_module("cognee.api.v1.visualize.get_schema_inventory")
router_module = import_module("cognee.api.v1.visualize.routers.get_schema_router")


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router_module.get_schema_router(), prefix="/api/v1/schema")
    return app


def test_schema_inventory_has_openapi_response_schema():
    schema = _app().openapi()
    operation = schema["paths"]["/api/v1/schema/inventory"]["get"]
    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    components = schema["components"]["schemas"]

    assert response_schema["type"] == "array"
    assert response_schema["items"]["$ref"].endswith("/SchemaInventoryItem")
    assert "SchemaInventoryItem" in components
    assert "SchemaInventoryRelationship" in components
    assert "relationships" in components["SchemaInventoryItem"]["properties"]


def test_schema_inventory_uses_generic_error_response(monkeypatch):
    app = _app()
    user_id = uuid4()

    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=user_id,
        tenant_id=None,
    )

    async def fake_authorized_datasets(dataset_ids, _permission, _user):
        return [SimpleNamespace(id=dataset_ids[0])]

    async def fail_inventory(**_kwargs):
        raise RuntimeError("database path /secret/internal leaked")

    monkeypatch.setattr(router_module, "send_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        router_module,
        "get_authorized_existing_datasets",
        fake_authorized_datasets,
    )
    monkeypatch.setattr(inventory_module, "get_schema_inventory", fail_inventory)

    response = TestClient(app).get(
        "/api/v1/schema/inventory",
        params={"dataset_id": str(uuid4())},
    )

    assert response.status_code == 409
    assert response.json() == {"error": "Failed to build schema inventory"}
