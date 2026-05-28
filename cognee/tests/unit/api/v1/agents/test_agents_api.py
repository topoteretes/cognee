from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cognee.api.v1.agents import get_agents_router
from cognee.modules.agents.registry import clear_registered_agent_connections
from cognee.modules.users.methods import get_authenticated_user


def _patch_operations(monkeypatch, user, datasets=None):
    async def readable_datasets_for(_user):
        return datasets or []

    async def visible_user_ids(_user):
        return [user.id]

    async def trace_agents_for_user(**_kwargs):
        return []

    async def persisted_agent_connections(_user_id):
        return []

    monkeypatch.setattr(
        "cognee.modules.agents.operations._readable_datasets_for",
        readable_datasets_for,
    )
    monkeypatch.setattr("cognee.modules.agents.operations._visible_user_ids", visible_user_ids)
    monkeypatch.setattr(
        "cognee.modules.agents.operations._trace_agents_for_user",
        trace_agents_for_user,
    )
    monkeypatch.setattr(
        "cognee.modules.agents.operations.list_persisted_agent_connections",
        persisted_agent_connections,
    )

    async def noop_persist(_user_id, _connection):
        pass

    monkeypatch.setattr(
        "cognee.modules.agents.registry._persist_agent_connection",
        noop_persist,
    )


def test_agents_register_and_list_endpoint(monkeypatch):
    clear_registered_agent_connections()
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    dataset_id = uuid4()
    dataset = SimpleNamespace(
        id=dataset_id,
        name="company_brain",
        owner_id=user.id,
        tenant_id=user.tenant_id,
    )

    _patch_operations(monkeypatch, user, datasets=[dataset])

    app = FastAPI()
    app.include_router(get_agents_router(), prefix="/api/v1/agents")
    app.dependency_overrides[get_authenticated_user] = lambda: user

    with TestClient(app) as client:
        created = client.post(
            "/api/v1/agents/register",
            json={
                "name": "support_agent",
                "type": "api",
                "memory_mode": "hybrid",
                "session_id": "support-prod",
                "dataset_ids": [str(dataset_id)],
                "source": "api",
            },
        )
        assert created.status_code == 201
        assert created.json()["name"] == "support_agent"

        response = client.get("/api/v1/agents/connections")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["agents"][0]["name"] == "support_agent"
        assert body["agents"][0]["datasets"][0]["id"] == str(dataset_id)
        assert body["memory_sources"][0]["type"] == "company_brain"
        assert body["memory_sources"][0]["connected_agent_ids"] == [created.json()["id"]]


def test_agents_detail_endpoint_returns_404_for_unknown_agent(monkeypatch):
    clear_registered_agent_connections()
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())

    _patch_operations(monkeypatch, user)

    app = FastAPI()
    app.include_router(get_agents_router(), prefix="/api/v1/agents")
    app.dependency_overrides[get_authenticated_user] = lambda: user

    with TestClient(app) as client:
        response = client.get("/api/v1/agents/connections/missing")
        assert response.status_code == 404
