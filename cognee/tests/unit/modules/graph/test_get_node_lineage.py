"""Unit tests for the provenance lineage query API.

These use a small fake graph engine so they stay offline and deterministic. The
fake returns connections in the same shape as ``get_connections``: a list of
``(node, edge, neighbor)`` tuples where the edge dict carries ``relationship_name``,
``source_node_id`` and ``target_node_id``.
"""

import pytest

from cognee.modules.graph.methods.get_node_lineage import (
    get_derived_nodes,
    get_source_lineage,
)
from cognee.tasks.storage.provenance_lineage import (
    DERIVED_FROM_RELATIONSHIP,
    IN_DATASET_RELATIONSHIP,
)


def _node(node_id, name, node_type):
    return {"id": node_id, "name": name, "type": node_type}


def _edge(source_id, target_id, relationship_name):
    return {
        "relationship_name": relationship_name,
        "source_node_id": source_id,
        "target_node_id": target_id,
    }


class FakeGraphEngine:
    """Returns canned connections keyed by the queried node id.

    ``connections[node_id]`` is a list of ``(node, edge, neighbor)`` tuples, with
    the queried node as the first element (matching the default backend).
    """

    def __init__(self, connections):
        self.connections = connections

    async def get_connections(self, node_id):
        return self.connections.get(str(node_id), [])


# A small graph:
#   alice -derived_from-> doc1 -in_dataset-> ds
#   bob   -derived_from-> doc1
# alice is a shared entity that also appears in doc2 -in_dataset-> ds.
DS = _node("ds", "fleet_ops", "DatasetNode")
DOC1 = _node("doc1", "a.txt", "TextDocument")
DOC2 = _node("doc2", "b.txt", "TextDocument")
ALICE = _node("alice", "Alice", "Entity")
BOB = _node("bob", "Bob", "Entity")


def _engine():
    return FakeGraphEngine(
        {
            "alice": [
                (ALICE, _edge("alice", "doc1", DERIVED_FROM_RELATIONSHIP), DOC1),
                (ALICE, _edge("alice", "doc2", DERIVED_FROM_RELATIONSHIP), DOC2),
            ],
            "bob": [
                (BOB, _edge("bob", "doc1", DERIVED_FROM_RELATIONSHIP), DOC1),
            ],
            "doc1": [
                (DOC1, _edge("alice", "doc1", DERIVED_FROM_RELATIONSHIP), ALICE),
                (DOC1, _edge("bob", "doc1", DERIVED_FROM_RELATIONSHIP), BOB),
                (DOC1, _edge("doc1", "ds", IN_DATASET_RELATIONSHIP), DS),
            ],
            "doc2": [
                (DOC2, _edge("alice", "doc2", DERIVED_FROM_RELATIONSHIP), ALICE),
                (DOC2, _edge("doc2", "ds", IN_DATASET_RELATIONSHIP), DS),
            ],
            "ds": [
                (DS, _edge("doc1", "ds", IN_DATASET_RELATIONSHIP), DOC1),
                (DS, _edge("doc2", "ds", IN_DATASET_RELATIONSHIP), DOC2),
            ],
        }
    )


@pytest.mark.asyncio
async def test_source_lineage_returns_documents_and_datasets():
    lineage = await get_source_lineage("alice", graph_engine=_engine())

    assert {d["id"] for d in lineage["documents"]} == {"doc1", "doc2"}
    assert {d["id"] for d in lineage["datasets"]} == {"ds"}


@pytest.mark.asyncio
async def test_source_lineage_deduplicates_shared_dataset():
    """alice derives from two documents in the same dataset; ds appears once."""
    lineage = await get_source_lineage("alice", graph_engine=_engine())
    assert len(lineage["datasets"]) == 1


@pytest.mark.asyncio
async def test_source_lineage_empty_for_node_without_lineage():
    lineage = await get_source_lineage("unknown", graph_engine=_engine())
    assert lineage == {"documents": [], "datasets": []}


@pytest.mark.asyncio
async def test_derived_nodes_for_document():
    derived = await get_derived_nodes("doc1", graph_engine=_engine())
    assert {n["id"] for n in derived} == {"alice", "bob"}


@pytest.mark.asyncio
async def test_derived_nodes_for_dataset_returns_documents():
    derived = await get_derived_nodes("ds", graph_engine=_engine())
    assert {n["id"] for n in derived} == {"doc1", "doc2"}


@pytest.mark.asyncio
async def test_derived_nodes_ignores_outgoing_edges():
    """Querying a leaf entity yields nothing (its derived_from is outgoing)."""
    derived = await get_derived_nodes("alice", graph_engine=_engine())
    assert derived == []
