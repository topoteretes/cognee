"""Concurrent writes to a dataset's schema must serialize per dataset.

PATCH /datasets/{id}/schema reads the stored spec, merges in memory and writes the
merged document back; PUT reads the row to decide between insert and update. Two of
them interleaving on one dataset lose a change silently or collide on the row's
unique dataset_id. The router holds one in-process lock per dataset around both.
"""

import asyncio
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI

router_module = import_module("cognee.api.v1.datasets.routers.get_datasets_router")


class _InterleavingStore:
    """In-memory dataset_configurations table that behaves like two database sessions.

    ``arrived`` is set once every request of the test has passed the permission
    check, which runs before the lock. A read, and the window between a store's
    SELECT and its INSERT, wait for it before going on. Without the lock all
    requests are then inside the handler at once and interleave every time; with
    it they still all arrive, so the lock holder is never kept waiting, and the
    others queue behind it. No timing is involved.
    """

    def __init__(self):
        self.rows: dict = {}
        self.arrived = asyncio.Event()

    async def read(self, dataset_id):
        # The SELECT happens now; the session hands the row over once the other
        # requests are in flight, as overlapping requests do.
        row = dict(self.rows.get(dataset_id, {"graph_schema": None, "custom_prompt": None}))
        await self.arrived.wait()
        return row

    async def store(self, dataset_id, graph_schema, custom_prompt):
        existed = dataset_id in self.rows
        if not existed:
            await self.arrived.wait()  # the window between SELECT and INSERT
            if dataset_id in self.rows:
                raise RuntimeError("UNIQUE constraint failed: dataset_configurations.dataset_id")
        row = self.rows.setdefault(dataset_id, {"graph_schema": None, "custom_prompt": None})
        if graph_schema is not None:
            row["graph_schema"] = graph_schema
        if custom_prompt is not None:
            row["custom_prompt"] = custom_prompt


@pytest.fixture
def schema_app(monkeypatch, request):
    """``request.param`` is the number of concurrent requests the test sends."""
    dataset_id = uuid4()
    store = _InterleavingStore()
    expected = request.param
    checks = 0

    async def authorized(_dataset_ids, _permission, _user):
        nonlocal checks
        checks += 1
        if checks == expected:
            store.arrived.set()
        return [SimpleNamespace(id=dataset_id)]

    monkeypatch.setattr(router_module, "get_authorized_existing_datasets", authorized)
    monkeypatch.setattr(router_module, "_read_dataset_schema", store.read)
    monkeypatch.setattr(router_module, "_store_dataset_schema", store.store)

    app = FastAPI()
    app.include_router(router_module.get_datasets_router(), prefix="/api/v1/datasets")
    app.dependency_overrides[router_module.get_authenticated_user] = lambda: SimpleNamespace(
        id=uuid4(), tenant_id=None
    )
    return app, dataset_id, store


def _entity(name):
    return {"name": name, "fields": [{"name": "title", "kind": "primitive"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_app", [6], indirect=True)
async def test_concurrent_patches_all_land(schema_app):
    """Six patches each add one entity to a stored schema; none may be lost."""
    app, dataset_id, store = schema_app
    store.rows[dataset_id] = {
        "graph_schema": {"entities": [_entity("Base")]},
        "custom_prompt": None,
    }
    names = [f"Entity{i}" for i in range(6)]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            *(
                client.patch(
                    f"/api/v1/datasets/{dataset_id}/schema", json={"entities": [_entity(n)]}
                )
                for n in names
            )
        )

    assert [r.status_code for r in responses] == [200] * len(names)
    stored = [e["name"] for e in store.rows[dataset_id]["graph_schema"]["entities"]]
    assert sorted(stored) == sorted(["Base", *names]), "every concurrent patch must survive"
    for name, response in zip(names, responses):
        returned = [e["name"] for e in response.json()["graph_schema"]["entities"]]
        assert name in returned, "a writer's response must contain its own change"


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_app", [2], indirect=True)
async def test_concurrent_first_puts_do_not_collide_on_insert(schema_app):
    """With no row yet, both writers would take the insert branch; the second must
    not hit the unique dataset_id index."""
    app, dataset_id, store = schema_app
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        responses = await asyncio.gather(
            client.put(f"/api/v1/datasets/{dataset_id}/schema", json={"customPrompt": "one"}),
            client.put(f"/api/v1/datasets/{dataset_id}/schema", json={"customPrompt": "two"}),
        )

    assert [r.status_code for r in responses] == [200, 200]
    assert store.rows[dataset_id]["custom_prompt"] in {"one", "two"}
