from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key

provenance_module = importlib.import_module("cognee.api.v1.visualize.memory_provenance")


class _EmptyExecuteResult:
    def scalars(self):
        return self

    def unique(self):
        return self

    def all(self):
        return []


class _EmptySession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def execute(self, _stmt):
        return _EmptyExecuteResult()


class _EmptyRelationalEngine:
    def get_async_session(self):
        return _EmptySession()


def _patch_empty_relational(monkeypatch):
    import cognee.infrastructure.databases.relational as relational_module

    monkeypatch.setattr(
        relational_module,
        "get_relational_engine",
        lambda: _EmptyRelationalEngine(),
    )


@pytest.mark.asyncio
async def test_read_memory_graph_provenance_builds_memory_payload(monkeypatch):
    dataset_id = uuid4()
    data_id = uuid4()
    source_ref = make_source_ref_key(dataset_id, data_id)
    edge = EdgeIdentity("node-a", "node-b", "related")
    graph = SimpleNamespace(
        find_node_source_refs_by_dataset=AsyncMock(return_value={"node-a": [source_ref]}),
        find_edge_source_refs_by_dataset=AsyncMock(return_value={edge: [source_ref]}),
        get_graph_data=AsyncMock(
            return_value=(
                [
                    ("node-a", {"type": "Entity", "name": "Alice"}),
                    ("node-b", {"type": "Entity", "name": "Bob"}),
                ],
                [("node-a", "node-b", "related", {"weight": 1})],
            )
        ),
    )
    import cognee.infrastructure.databases.provenance.markers as markers_module
    import cognee.infrastructure.databases.unified as unified_module

    monkeypatch.setattr(
        unified_module,
        "get_unified_engine",
        AsyncMock(return_value=SimpleNamespace(graph=graph)),
    )
    monkeypatch.setattr(markers_module, "stores_provenance_in_graph", AsyncMock(return_value=True))

    payload = await provenance_module._read_memory_graph_provenance(dataset_ids=[str(dataset_id)])

    assert sorted(payload["nodes"]) == [
        provenance_module.Node("node-a", {"type": "Entity", "name": "Alice"}),
        provenance_module.Node("node-b", {"type": "Entity", "name": "Bob"}),
    ]
    assert payload["edges"] == [
        provenance_module.EdgeData("node-a", "node-b", "related", {"weight": 1})
    ]
    assert payload["links"] == [
        {"node_id": "node-a", "data_id": str(data_id), "dataset_id": str(dataset_id)}
    ]


@pytest.mark.asyncio
async def test_read_memory_graph_provenance_returns_none_for_unmarked_graph(monkeypatch):
    dataset_id = uuid4()
    graph = SimpleNamespace()
    import cognee.infrastructure.databases.provenance.markers as markers_module
    import cognee.infrastructure.databases.unified as unified_module

    monkeypatch.setattr(
        unified_module,
        "get_unified_engine",
        AsyncMock(return_value=SimpleNamespace(graph=graph)),
    )
    monkeypatch.setattr(markers_module, "stores_provenance_in_graph", AsyncMock(return_value=False))

    assert (
        await provenance_module._read_memory_graph_provenance(dataset_ids=[str(dataset_id)]) is None
    )


@pytest.mark.asyncio
async def test_read_memory_graph_provenance_surfaces_graph_read_error(monkeypatch):
    dataset_id = uuid4()
    data_id = uuid4()
    source_ref = make_source_ref_key(dataset_id, data_id)
    graph = SimpleNamespace(
        find_node_source_refs_by_dataset=AsyncMock(return_value={"node-a": [source_ref]}),
        find_edge_source_refs_by_dataset=AsyncMock(return_value={}),
        get_graph_data=AsyncMock(side_effect=RuntimeError("graph read failed")),
    )
    import cognee.infrastructure.databases.provenance.markers as markers_module
    import cognee.infrastructure.databases.unified as unified_module

    monkeypatch.setattr(
        unified_module,
        "get_unified_engine",
        AsyncMock(return_value=SimpleNamespace(graph=graph)),
    )
    monkeypatch.setattr(markers_module, "stores_provenance_in_graph", AsyncMock(return_value=True))

    with pytest.raises(RuntimeError, match="graph read failed"):
        await provenance_module._read_memory_graph_provenance(dataset_ids=[str(dataset_id)])


@pytest.mark.asyncio
async def test_get_memory_provenance_graph_falls_back_when_graph_reader_returns_none(monkeypatch):
    _patch_empty_relational(monkeypatch)
    graph_reader = AsyncMock(return_value=None)
    relational_reader = AsyncMock(
        return_value={
            "nodes": [("memory-node", {"type": "Entity", "name": "Fallback"})],
            "edges": [],
            "links": [],
        }
    )
    monkeypatch.setattr(provenance_module, "_read_agents", AsyncMock(return_value=[]))
    monkeypatch.setattr(provenance_module, "_read_sessions", AsyncMock(return_value=[]))
    monkeypatch.setattr(provenance_module, "_read_memory_graph_provenance", graph_reader)
    monkeypatch.setattr(provenance_module, "_read_memory_relational", relational_reader)

    nodes, _edges = await provenance_module.get_memory_provenance_graph(include_memory=True)

    assert provenance_module.Node("memory-node", {"type": "Entity", "name": "Fallback"}) in nodes
    graph_reader.assert_awaited_once_with(dataset_ids=[])
    relational_reader.assert_awaited_once_with(dataset_ids=None)


@pytest.mark.asyncio
async def test_get_memory_provenance_graph_does_not_fallback_after_graph_error(monkeypatch):
    _patch_empty_relational(monkeypatch)
    graph_reader = AsyncMock(side_effect=RuntimeError("graph read failed"))
    relational_reader = AsyncMock()
    monkeypatch.setattr(provenance_module, "_read_agents", AsyncMock(return_value=[]))
    monkeypatch.setattr(provenance_module, "_read_sessions", AsyncMock(return_value=[]))
    monkeypatch.setattr(provenance_module, "_read_memory_graph_provenance", graph_reader)
    monkeypatch.setattr(provenance_module, "_read_memory_relational", relational_reader)

    with pytest.raises(RuntimeError, match="graph read failed"):
        await provenance_module.get_memory_provenance_graph(include_memory=True)

    graph_reader.assert_awaited_once_with(dataset_ids=[])
    relational_reader.assert_not_called()
