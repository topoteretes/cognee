from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
import importlib

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key

provenance_module = importlib.import_module("cognee.api.v1.visualize.memory_provenance")


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
