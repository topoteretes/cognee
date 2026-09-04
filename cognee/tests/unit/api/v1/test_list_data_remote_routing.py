"""datasets.list_data() must read from the remote instance once serve() is
connected — the dataset lives on the server, so the local store would report
it missing or empty."""

import importlib
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from cognee.api.v1.serve import state as state_mod
from cognee.api.v1.serve.cloud_client import CloudClient

# ``cognee.api.v1.datasets`` re-exports the ``datasets`` class under the same
# name as its module; resolve the module explicitly to patch its globals.
datasets_mod = importlib.import_module("cognee.api.v1.datasets.datasets")


def _server_row(data_id: UUID, dataset_id: UUID) -> dict:
    """One row as GET /api/v1/datasets/{id}/data returns it (camelCase DataDTO)."""
    return {
        "id": str(data_id),
        "name": "note.md",
        "createdAt": "2026-09-04T10:00:00Z",
        "updatedAt": None,
        "extension": "md",
        "mimeType": "text/markdown",
        "rawDataLocation": "/srv/data/note.md",
        "datasetId": str(dataset_id),
        "label": "notes",
        "externalMetadata": {"source": "test"},
    }


class _StubClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def list_data(self, dataset_id):
        self.calls.append(dataset_id)
        return self.rows


@pytest.mark.asyncio
async def test_list_data_routes_to_remote_and_returns_local_shaped_rows(monkeypatch):
    data_id, dataset_id = uuid4(), uuid4()
    stub = _StubClient([_server_row(data_id, dataset_id)])
    monkeypatch.setattr(state_mod, "_remote_client", stub)

    async def local_path_reached():
        raise AssertionError("list_data() did local work while a remote client was connected")

    monkeypatch.setattr(datasets_mod, "get_default_user", local_path_reached)

    rows = await datasets_mod.datasets.list_data(dataset_id)

    assert stub.calls == [dataset_id]
    assert len(rows) == 1
    row = rows[0]
    # Same attribute names and types callers get from local Data rows.
    assert row.id == data_id and isinstance(row.id, UUID)
    assert row.dataset_id == dataset_id
    assert row.name == "note.md"
    assert row.mime_type == "text/markdown"
    assert row.raw_data_location == "/srv/data/note.md"
    assert isinstance(row.created_at, datetime)
    assert row.updated_at is None
    assert row.label == "notes"
    assert row.external_metadata == {"source": "test"}


@pytest.mark.asyncio
async def test_list_data_remote_keeps_unknown_server_fields(monkeypatch):
    data_id, dataset_id = uuid4(), uuid4()
    row = {**_server_row(data_id, dataset_id), "tokenCount": 42}
    monkeypatch.setattr(state_mod, "_remote_client", _StubClient([row]))

    rows = await datasets_mod.datasets.list_data(dataset_id)

    assert rows[0].tokenCount == 42


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload if payload is not None else []
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


def _client_with_fake_get(monkeypatch, response):
    client = CloudClient("http://remote.invalid", "key")
    captured = {}

    @asynccontextmanager
    async def fake_get(url, **kwargs):
        captured["url"] = url
        yield response

    session = MagicMock()
    session.get = fake_get

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    return client, captured


@pytest.mark.asyncio
async def test_cloud_client_list_data_hits_the_dataset_data_route(monkeypatch):
    dataset_id = uuid4()
    client, captured = _client_with_fake_get(monkeypatch, _FakeResponse(payload=[{"id": "x"}]))

    rows = await client.list_data(dataset_id)

    assert rows == [{"id": "x"}]
    assert captured["url"] == f"http://remote.invalid/api/v1/datasets/{dataset_id}/data"


@pytest.mark.asyncio
async def test_cloud_client_list_data_surfaces_remote_errors(monkeypatch):
    client, _ = _client_with_fake_get(
        monkeypatch, _FakeResponse(status=404, text='{"error":"no such dataset"}')
    )

    with pytest.raises(RuntimeError, match=r"Remote list_data failed \(404\)"):
        await client.list_data(uuid4())
