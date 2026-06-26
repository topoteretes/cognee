from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    EdgeIdentity,
    make_source_ref_key,
    make_source_run_ref,
)
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import _migrate_graph


class _Graph:
    def __init__(self, node_snapshot, edge_snapshot):
        self.node_snapshot = node_snapshot
        self.edge_snapshot = edge_snapshot
        self.node_attaches = []
        self.edge_attaches = []
        self.added_nodes = []
        self.added_edges = []
        self.deleted_nodes = []

    async def get_node_delete_data(self, node_ids):
        return {node_id: self.node_snapshot for node_id in node_ids}

    async def get_edge_delete_data(self, edges):
        return {edge: self.edge_snapshot for edge in edges}

    async def add_nodes(self, nodes):
        self.added_nodes.extend(nodes)

    async def add_edges(self, edges):
        self.added_edges.extend(edges)

    async def delete_nodes(self, node_ids):
        self.deleted_nodes.extend(node_ids)

    async def attach_node_source_refs(self, node_ids, refs, pipeline_run_id=None):
        self.node_attaches.append((node_ids, refs, pipeline_run_id))

    async def attach_edge_source_refs(self, edges, refs, pipeline_run_id=None):
        self.edge_attaches.append((edges, refs, pipeline_run_id))


@pytest.mark.asyncio
async def test_migrate_graph_preserves_existing_graph_provenance():
    dataset_id = uuid4()
    data_id = uuid4()
    run_id = uuid4()
    source_ref = make_source_ref_key(dataset_id, data_id)
    source_run_ref = make_source_run_ref(run_id, source_ref)
    node_snapshot = SimpleNamespace(source_ref_keys=[source_ref], source_run_refs=[source_run_ref])
    edge_snapshot = SimpleNamespace(source_ref_keys=[source_ref], source_run_refs=[source_run_ref])
    graph = _Graph(node_snapshot, edge_snapshot)

    remapped = await _migrate_graph(
        graph,
        {"old-node": "new-node"},
        {"old-node": {"id": "old-node", "name": "Alice", "type": "Entity"}},
        [("old-node", "chunk", "contains", {"source_node_id": "old-node"})],
    )

    assert remapped == 1
    assert graph.node_attaches == [(["new-node"], [source_ref], str(run_id))]
    assert graph.edge_attaches == [
        ([EdgeIdentity("new-node", "chunk", "contains")], [source_ref], str(run_id))
    ]
