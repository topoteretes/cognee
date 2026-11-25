"""The cloud client's conflict calls speak the conflicts router's wire format."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from cognee.api.v1.serve.cloud_client import CloudClient
from cognee.modules.conflicts.models import ConflictDetail


def _server_conflict(conflict_id, dataset_id, fact_id) -> dict:
    """One conflict as GET /api/v1/conflicts returns it (camelCase ConflictDetail)."""
    return {
        "id": str(conflict_id),
        "datasetId": str(dataset_id),
        "aboutId": "company",
        "attribute": "has_ceo",
        "kind": "fixed",
        "status": "resolved",
        "text": "Resolved by user: Alice is CEO.",
        "facts": [
            {
                "id": str(fact_id),
                "text": "Alice is CEO",
                "sourceId": "company",
                "targetId": "alice",
                "relationship": "has_ceo",
                "status": "current",
                "sources": [],
            }
        ],
        "resolutionSource": "user",
        "selectedFactId": str(fact_id),
        "selectedFact": "Alice is CEO",
    }


class _FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status = status
        self._payload = payload
        self._text = text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text


def _client(monkeypatch, response):
    client = CloudClient("http://remote.invalid", "key")
    captured = {}

    @asynccontextmanager
    async def fake_request(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        yield response

    session = MagicMock()
    session.get = fake_request
    session.post = fake_request

    async def get_session():
        return session

    monkeypatch.setattr(client, "_get_session", get_session)
    return client, captured


@pytest.mark.asyncio
async def test_list_conflicts_reads_the_shared_wire_model(monkeypatch):
    conflict_id, dataset_id, fact_id = uuid4(), uuid4(), uuid4()
    client, captured = _client(
        monkeypatch, _FakeResponse(payload=[_server_conflict(conflict_id, dataset_id, fact_id)])
    )
    rows = await client.list_conflicts(dataset_id, status="resolved")
    assert captured["url"] == "http://remote.invalid/api/v1/conflicts"
    assert captured["params"] == {"dataset_id": str(dataset_id), "status": "resolved"}
    assert isinstance(rows[0], ConflictDetail)
    assert rows[0].id == conflict_id and rows[0].selected_fact_id == fact_id
    assert rows[0].facts[0].id == fact_id and rows[0].facts[0].status == "current"

    client, captured = _client(monkeypatch, _FakeResponse(payload=[]))
    assert await client.list_conflicts(dataset_id) == []
    assert captured["params"] == {"dataset_id": str(dataset_id)}

    client, _ = _client(monkeypatch, _FakeResponse(payload={"detail": "not a list"}))
    with pytest.raises(RuntimeError, match="invalid conflict list"):
        await client.list_conflicts(dataset_id)


@pytest.mark.asyncio
async def test_resolve_conflict_posts_the_choice_and_surfaces_server_errors(monkeypatch):
    conflict_id, dataset_id, fact_id = uuid4(), uuid4(), uuid4()
    client, captured = _client(
        monkeypatch, _FakeResponse(payload=_server_conflict(conflict_id, dataset_id, fact_id))
    )
    result = await client.resolve_conflict(
        conflict_id, dataset_id=dataset_id, selected_fact_id=fact_id, reason="Verified"
    )
    assert captured["url"] == f"http://remote.invalid/api/v1/conflicts/{conflict_id}/resolve"
    assert captured["json"] == {
        "datasetId": str(dataset_id),
        "selectedFactId": str(fact_id),
        "reason": "Verified",
    }
    assert isinstance(result, ConflictDetail) and result.selected_fact == "Alice is CEO"

    client, captured = _client(monkeypatch, _FakeResponse(status=422, text="fact not in conflict"))
    with pytest.raises(RuntimeError, match="422"):
        await client.resolve_conflict(conflict_id, dataset_id=dataset_id, selected_fact_id=fact_id)
    assert captured["json"] == {"datasetId": str(dataset_id), "selectedFactId": str(fact_id)}
