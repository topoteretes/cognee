import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cognee.modules.conflicts.models import ConflictDetail

router_module = importlib.import_module("cognee.api.v1.conflicts.routers.get_conflicts_router")


@pytest.mark.asyncio
async def test_http_lists_and_resolves_with_authenticated_user_and_typed_body(monkeypatch):
    user = SimpleNamespace(id=uuid4())
    dataset_id, conflict_id, fact_id = uuid4(), uuid4(), uuid4()
    result = ConflictDetail(
        id=conflict_id,
        dataset_id=dataset_id,
        about_id=str(uuid4()),
        attribute="has_ceo",
        kind="fixed",
        status="resolved",
        text="Alice is CEO",
        selected_fact_id=fact_id,
        selected_fact="Alice is CEO",
        resolution_source="user",
    )
    resolve = AsyncMock(return_value=result)
    listing = AsyncMock(return_value=[result])
    monkeypatch.setattr(router_module, "resolve_conflict", resolve)
    monkeypatch.setattr(router_module, "list_conflicts", listing)
    app = FastAPI()
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: user
    app.include_router(router_module.get_conflicts_router(), prefix="/api/v1/conflicts")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(f"/api/v1/conflicts?dataset_id={dataset_id}&status=resolved")
        assert response.status_code == 200 and response.json()[0]["selectedFactId"] == str(fact_id)
        response = await client.post(
            f"/api/v1/conflicts/{conflict_id}/resolve",
            json={
                "datasetId": str(dataset_id),
                "selectedFactId": str(fact_id),
                "reason": "Verified",
            },
        )
        assert response.status_code == 200 and response.json()["status"] == "resolved"
        resolve.assert_awaited_once_with(
            conflict_id,
            dataset_id=dataset_id,
            selected_fact_id=fact_id,
            reason="Verified",
            user=user,
        )
        bad = await client.post(
            f"/api/v1/conflicts/{conflict_id}/resolve",
            json={"dataset_id": str(dataset_id), "selected_fact_id": "not-a-uuid"},
        )
        assert bad.status_code == 422
    listing.assert_awaited_once_with(dataset_id, status="resolved", user=user)
