"""Retiring chunks must cost the graph a constant number of scans.

``find_nodes_by_source_ref`` / ``find_edges_by_source_ref`` are unindexed
``... CONTAINS $token`` scans. Issuing them per deleted chunk made a region
replacement cost two full graph scans for every chunk it retired — the one
place where this feature's cost grew with GRAPH size rather than EDIT size,
which is precisely the property it exists to deliver.
"""

import asyncio
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.provenance import (
    make_chunk_source_ref_key,
    make_source_ref_key,
)
from cognee.modules.graph.methods.delete_chunks_incremental import _strip_dead_chunk_refs


class _RecordingGraph:
    """Counts scans and removals, and applies removals so state is checkable."""

    def __init__(self, refs_by_node=None, refs_by_edge=None):
        self.refs_by_node = refs_by_node or {}
        self.refs_by_edge = refs_by_edge or {}
        self.per_ref_scans = 0
        self.per_dataset_scans = 0
        self.removal_calls = 0

    async def find_nodes_by_source_ref(self, _key):
        self.per_ref_scans += 1
        return []

    async def find_edges_by_source_ref(self, _key):
        self.per_ref_scans += 1
        return []

    async def find_node_source_refs_by_dataset(self, _dataset_id):
        self.per_dataset_scans += 1
        return {node: list(refs) for node, refs in self.refs_by_node.items()}

    async def find_edge_source_refs_by_dataset(self, _dataset_id):
        self.per_dataset_scans += 1
        return {edge: list(refs) for edge, refs in self.refs_by_edge.items()}

    async def remove_node_source_refs(self, holders, keys):
        self.removal_calls += 1
        for holder in holders:
            self.refs_by_node[holder] = [
                ref for ref in self.refs_by_node[holder] if ref not in keys
            ]

    async def remove_edge_source_refs(self, holders, keys):
        self.removal_calls += 1
        for holder in holders:
            self.refs_by_edge[holder] = [
                ref for ref in self.refs_by_edge[holder] if ref not in keys
            ]


def _run(graph, deleting, dataset_id, data_id, monkeypatch):
    import cognee.infrastructure.databases.provenance.markers as markers

    async def _stores(_engine):
        return True

    monkeypatch.setattr(markers, "stores_provenance_in_graph", _stores)
    asyncio.run(_strip_dead_chunk_refs(graph, deleting, dataset_id, data_id))


def test_scan_count_is_constant_in_the_number_of_deleted_chunks(monkeypatch):
    dataset_id, data_id = uuid4(), uuid4()
    chunk_ids = [uuid4() for _ in range(20)]
    doc_key = make_source_ref_key(dataset_id, data_id)

    refs_by_node = {
        f"node-{index}": [doc_key, make_chunk_source_ref_key(dataset_id, data_id, chunk_id)]
        for index, chunk_id in enumerate(chunk_ids)
    }
    graph = _RecordingGraph(refs_by_node=refs_by_node)

    _run(graph, {str(c) for c in chunk_ids}, dataset_id, data_id, monkeypatch)

    assert graph.per_ref_scans == 0, "per-chunk scans must be gone"
    assert graph.per_dataset_scans == 2, (
        f"expected one node scan and one edge scan, got {graph.per_dataset_scans}"
    )


def test_surviving_artifacts_keep_their_document_key(monkeypatch):
    """Stripping a dead chunk's key must not strip the document-scoped one.

    An artifact left with no refs at all is invisible to delete_by_document,
    which resolves through the dataset's ref maps — so it would leak.
    """
    dataset_id, data_id = uuid4(), uuid4()
    dead, alive = uuid4(), uuid4()
    doc_key = make_source_ref_key(dataset_id, data_id)
    dead_key = make_chunk_source_ref_key(dataset_id, data_id, dead)
    alive_key = make_chunk_source_ref_key(dataset_id, data_id, alive)

    graph = _RecordingGraph(
        refs_by_node={
            "shared": [doc_key, dead_key, alive_key],
            "only-dead": [dead_key],
            "untouched": [doc_key],
        }
    )

    _run(graph, {str(dead)}, dataset_id, data_id, monkeypatch)

    assert graph.refs_by_node["shared"] == [doc_key, alive_key]
    assert graph.refs_by_node["untouched"] == [doc_key]
    # An artifact owned ONLY by the dead chunk is emptied here; the deletion
    # planner is what removes the node itself.
    assert graph.refs_by_node["only-dead"] == []


def test_artifacts_owned_by_several_dead_chunks_are_rewritten_once(monkeypatch):
    """Read-modify-write per artifact, not per dead key."""
    dataset_id, data_id = uuid4(), uuid4()
    dead = [uuid4() for _ in range(4)]
    dead_keys = [make_chunk_source_ref_key(dataset_id, data_id, c) for c in dead]

    graph = _RecordingGraph(refs_by_node={"hub": list(dead_keys)})
    _run(graph, {str(c) for c in dead}, dataset_id, data_id, monkeypatch)

    assert graph.removal_calls == 1, "four dead keys on one artifact must be one call"
    assert graph.refs_by_node["hub"] == []


def test_nothing_to_strip_issues_no_removals(monkeypatch):
    dataset_id, data_id = uuid4(), uuid4()
    graph = _RecordingGraph(
        refs_by_node={"n": [make_source_ref_key(dataset_id, data_id)]},
    )

    _run(graph, {str(uuid4())}, dataset_id, data_id, monkeypatch)

    assert graph.removal_calls == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
