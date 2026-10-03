"""HTTP read surface of the audit ledger: dataset ACL + payload shapes.

Runs against the isolated ledger from ``conftest.manager`` through an in-process
ASGI transport (same event loop, so the aiosqlite engine is shared).
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

import cognee.api.v1.provenance.routers.get_provenance_router
from cognee.api.v1.provenance.routers import get_provenance_router
from cognee.modules.provenance.tombstones import (
    dataset_id_from_ledger_key,
    ledger_edge_key,
    ledger_node_key,
)
from cognee.modules.users.methods import get_authenticated_user

router_module = sys.modules["cognee.api.v1.provenance.routers.get_provenance_router"]

pytestmark = pytest.mark.asyncio


class TestKeyParsing:
    def test_node_edge_and_archive_keys(self):
        ds = uuid4()
        assert dataset_id_from_ledger_key(ledger_node_key(ds, "n1")) == ds
        assert dataset_id_from_ledger_key(ledger_edge_key(ds, "a", "b", "knows")) == ds
        assert (
            dataset_id_from_ledger_key(f"{ledger_node_key(ds, 'n1')}:v:2026-01-01T00:00:00") == ds
        )

    def test_unscoped_keys(self):
        assert dataset_id_from_ledger_key("custom-entity") is None
        assert dataset_id_from_ledger_key("rel:a:knows:b") is None
        assert dataset_id_from_ledger_key("") is None


@pytest_asyncio.fixture
async def app(manager, monkeypatch):
    ds_allowed, ds_denied = uuid4(), uuid4()

    async def _authorized(user, dataset_id, permission_type="read"):
        if dataset_id != ds_allowed:
            return None
        return SimpleNamespace(id=dataset_id, owner_id=uuid4())

    monkeypatch.setattr(router_module, "get_authorized_dataset", AsyncMock(side_effect=_authorized))

    app = FastAPI()
    app.include_router(get_provenance_router(), prefix="/api/v1/provenance")
    app.state.user = SimpleNamespace(id=uuid4(), email="u@x.io", is_superuser=False)

    async def _user():
        return app.state.user

    app.dependency_overrides[get_authenticated_user] = _user

    batch = manager.batch()
    batch.track_entity(ledger_node_key(ds_allowed, "n1"), source="doc-a")
    batch.track_entity(ledger_node_key(ds_allowed, "n2"), source="doc-a")
    batch.track_relationship(
        ledger_edge_key(ds_allowed, "n1", "n2", "knows"),
        source="doc-a",
        used_entities=[ledger_node_key(ds_allowed, "n1"), ledger_node_key(ds_allowed, "n2")],
    )
    batch.track_entity(ledger_node_key(ds_denied, "n1"), source="doc-b")
    batch.track_entity("unscoped-entity", source="custom")
    await batch.commit()
    await manager.invalidate(ledger_node_key(ds_allowed, "n2"), "auditor", reason="retracted")

    app.state.ds_allowed, app.state.ds_denied = ds_allowed, ds_denied
    return app


@pytest_asyncio.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        client.app = app
        yield client


class TestPerEntityReads:
    async def test_entry_lineage_history_for_readable_dataset(self, client):
        ds = client.app.state.ds_allowed
        edge_key = ledger_edge_key(ds, "n1", "n2", "knows")

        entry = await client.get(f"/api/v1/provenance/entry/{edge_key}")
        assert entry.status_code == 200
        assert entry.json()["entity_id"] == edge_key

        lineage = await client.get(f"/api/v1/provenance/lineage/{edge_key}")
        assert lineage.status_code == 200
        body = lineage.json()
        assert body["entity_count"] == 3
        assert body["integrity_verified"] is True

        history = await client.get(f"/api/v1/provenance/history/{ledger_node_key(ds, 'n2')}")
        assert history.status_code == 200
        versions = history.json()
        assert [v["version"] for v in versions] == [1, 2]
        assert versions[-1]["invalidated"] is True
        assert versions[-1]["invalidated_by"] == "auditor"

    async def test_unreadable_dataset_is_403_even_when_tracked(self, client):
        key = ledger_node_key(client.app.state.ds_denied, "n1")
        for route in ("entry", "lineage", "history"):
            response = await client.get(f"/api/v1/provenance/{route}/{key}")
            assert response.status_code == 403, route

    async def test_untracked_key_in_readable_dataset_is_404(self, client):
        key = ledger_node_key(client.app.state.ds_allowed, "nope")
        for route in ("entry", "lineage", "history"):
            response = await client.get(f"/api/v1/provenance/{route}/{key}")
            assert response.status_code == 404, route

    async def test_unscoped_keys_are_superuser_only(self, client):
        response = await client.get("/api/v1/provenance/entry/unscoped-entity")
        assert response.status_code == 403
        client.app.state.user.is_superuser = True
        response = await client.get("/api/v1/provenance/entry/unscoped-entity")
        assert response.status_code == 200


class TestLedgerWalks:
    async def test_scoped_walks_need_dataset_read_access(self, client):
        ds_ok, ds_no = client.app.state.ds_allowed, client.app.state.ds_denied
        for route in ("verify", "check", "statistics"):
            ok = await client.get(f"/api/v1/provenance/{route}", params={"dataset_id": str(ds_ok)})
            assert ok.status_code == 200, route
            assert ok.json()["dataset_id"] == str(ds_ok)
            denied = await client.get(
                f"/api/v1/provenance/{route}", params={"dataset_id": str(ds_no)}
            )
            assert denied.status_code == 403, route

    async def test_scoped_results_only_see_their_dataset(self, client):
        ds = client.app.state.ds_allowed
        stats = (
            await client.get("/api/v1/provenance/statistics", params={"dataset_id": str(ds)})
        ).json()
        assert stats["live_entries"] == 3
        assert stats["archived_entries"] == 1
        assert stats["invalidated_count"] == 1
        assert stats["entity_types"] == {"entity": 3, "relationship": 1}

        verify = (
            await client.get("/api/v1/provenance/verify", params={"dataset_id": str(ds)})
        ).json()
        assert verify == {
            "valid": True,
            "total_entries": 4,
            "broken_links": [],
            "dataset_id": str(ds),
        }

    async def test_drift_requires_dataset_and_read_access(self, client, monkeypatch):
        from cognee.modules.provenance.manager import ProvenanceManager

        calls = []

        async def _drift(self, dataset_id, owner_id=None):
            calls.append((dataset_id, owner_id))
            return {"valid": True, "dataset_id": str(dataset_id), "checked": 0}

        monkeypatch.setattr(ProvenanceManager, "check_drift", _drift)
        ds_ok, ds_no = client.app.state.ds_allowed, client.app.state.ds_denied

        assert (await client.get("/api/v1/provenance/drift")).status_code == 422
        denied = await client.get("/api/v1/provenance/drift", params={"dataset_id": str(ds_no)})
        assert denied.status_code == 403 and calls == []
        ok = await client.get("/api/v1/provenance/drift", params={"dataset_id": str(ds_ok)})
        assert ok.status_code == 200 and ok.json()["valid"] is True
        assert calls[0][0] == ds_ok

    async def test_whole_ledger_walks_are_superuser_only(self, client):
        for route in ("verify", "check", "statistics", "export"):
            response = await client.get(f"/api/v1/provenance/{route}")
            assert response.status_code == 403, route

        client.app.state.user.is_superuser = True
        stats = await client.get("/api/v1/provenance/statistics")
        assert stats.status_code == 200
        assert stats.json()["total_entries"] == 6
        assert stats.json()["dataset_id"] is None

    async def test_export_is_jsonl_in_chain_order(self, client):
        ds = client.app.state.ds_allowed
        response = await client.get("/api/v1/provenance/export", params={"dataset_id": str(ds)})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        rows = [json.loads(line) for line in response.text.splitlines()]
        assert len(rows) == 4
        sequence_ids = [row["sequence_id"] for row in rows]
        assert sequence_ids == sorted(sequence_ids)

        current = await client.get(
            "/api/v1/provenance/export",
            params={"dataset_id": str(ds), "include_archived": "false"},
        )
        assert len(current.text.splitlines()) == 3

    async def test_status_reports_the_flag(self, client, monkeypatch):
        import cognee.tasks.provenance.record_provenance

        record_module = sys.modules["cognee.tasks.provenance.record_provenance"]
        monkeypatch.setattr(record_module, "provenance_tracking_enabled", lambda: True)
        response = await client.get("/api/v1/provenance/status")
        assert response.json() == {"provenance_tracking": True, "anchoring_configured": False}
