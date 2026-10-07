"""With access control off every dataset writes into one graph, so a dataset's
view must keep only what that dataset owns: an entity another dataset also
mentions stays, an edge only another dataset drew does not."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from cognee.infrastructure.databases.graph.bounded_neighborhood import hop_distances
from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface
from cognee.infrastructure.databases.provenance import (
    GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
    GRAPH_DELETE_MODE_KEY,
    GRAPH_PROVENANCE_VERSION,
    GRAPH_PROVENANCE_VERSION_KEY,
)
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.modules.visualization import subgraph_data
from cognee.modules.visualization.graph_stream import stream_graph_events
from cognee.modules.visualization.subgraph_data import (
    OwnedGraph,
    fetch_visualization_graph_data,
    get_owned_graph,
)

# Dataset "a" owns a0..a2, "b" owns b0..b2, and both mention "shared".
NODE_OWNERS = {"a0": "a", "a1": "a", "a2": "a", "b0": "b", "b1": "b", "b2": "b", "shared": "ab"}
EDGE_OWNERS = {
    ("shared", "a0", "contains"): "a",
    ("a0", "a1", "related_to"): "a",
    ("a1", "a2", "related_to"): "a",
    ("a0", "a1", "mentions"): "b",  # drawn by b between two of a's nodes
    ("shared", "b0", "contains"): "b",
    ("b0", "b1", "related_to"): "b",
    ("b1", "b2", "related_to"): "b",
}
A_NODES = {node for node, owners in NODE_OWNERS.items() if "a" in owners}
A_EDGES = {edge for edge, owners in EDGE_OWNERS.items() if owners == "a"}


class SharedGraph:
    """Both datasets in one graph that stores its provenance in the graph."""

    def __init__(self):
        self.nodes = [(node, {"name": node, "type": "Entity"}) for node in NODE_OWNERS]
        self.edges = [(*edge, {}) for edge in EDGE_OWNERS]

    async def get_graph_metadata(self):
        return {
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
        }

    async def find_node_source_refs_by_dataset(self, dataset_id):
        return {node: ["ref"] for node, owners in NODE_OWNERS.items() if dataset_id in owners}

    async def find_edge_source_refs_by_dataset(self, dataset_id):
        return {
            EdgeIdentity(*edge): ["ref"]
            for edge, owners in EDGE_OWNERS.items()
            if dataset_id in owners
        }

    async def get_graph_data(self):
        return self.nodes, self.edges

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        reached = {
            node for node, hops in hop_distances(self.edges, node_ids).items() if hops <= depth
        }
        return (
            [node for node in self.nodes if node[0] in reached],
            [edge for edge in self.edges if edge[0] in reached and edge[1] in reached],
        )

    def iter_bounded_neighborhood(self, *args, **kwargs):
        return GraphDBInterface.iter_bounded_neighborhood(self, *args, **kwargs)

    async def get_entity_type_names(self, entity_ids):
        return {}


@pytest.fixture(autouse=True)
def access_control_off(monkeypatch):
    monkeypatch.setattr(subgraph_data, "backend_access_control_enabled", lambda: False)


def _ids(nodes):
    return {node for node, _ in nodes}


def _keys(edges):
    return {edge[:3] for edge in edges}


@pytest.mark.asyncio
async def test_a_dataset_view_holds_only_what_the_dataset_owns():
    graph = SharedGraph()
    owned = await get_owned_graph(graph, "a")

    for full in (False, True):
        nodes, edges = await fetch_visualization_graph_data(
            graph, owned=owned, full=full, seed_node_ids=["b1"]
        )
        assert _ids(nodes) == A_NODES
        assert _keys(edges) == A_EDGES


@pytest.mark.asyncio
async def test_the_node_budget_goes_to_the_datasets_own_nodes():
    graph = SharedGraph()
    owned = await get_owned_graph(graph, "a")

    nodes, _ = await fetch_visualization_graph_data(
        graph, owned=owned, seed_node_ids=["shared"], max_nodes=3
    )

    assert [node for node, _ in nodes] == ["shared", "a0", "a1"]


@pytest.mark.asyncio
async def test_a_streamed_view_sends_only_what_the_dataset_owns():
    graph = SharedGraph()
    owned = await get_owned_graph(graph, "a")

    events = [
        event
        async for event in stream_graph_events(
            graph,
            owned=owned,
            query=None,
            seed_node_ids=["b0", "a0"],
            neighborhood_depth=2,
            seed_top_k=10,
            max_nodes=100,
            chunk_size=2,
        )
    ]

    chunks = [data for name, data in events if name == "chunk"]
    assert events[0][1]["seeds"] == ["a0"]
    assert {node["id"] for chunk in chunks for node in chunk["nodes"]} == A_NODES
    assert {
        (link["source"], link["target"], link["relation"])
        for chunk in chunks
        for link in chunk["links"]
    } == A_EDGES


@pytest.mark.asyncio
async def test_a_ledger_graph_is_owned_by_its_ledger_nodes(monkeypatch):
    graph = MagicMock(spec=["get_graph_metadata"])

    async def unmarked():
        return {}

    graph.get_graph_metadata = unmarked
    session = MagicMock()

    async def scalars(statement):
        return ["a0", "a1"]

    session.scalars = scalars

    @asynccontextmanager
    async def get_async_session():
        yield session

    monkeypatch.setattr(
        subgraph_data,
        "get_relational_engine",
        lambda: MagicMock(get_async_session=get_async_session),
    )

    owned = await get_owned_graph(graph, "a")

    assert owned == OwnedGraph(nodes={"a0", "a1"}, edges=None)
    nodes, edges = subgraph_data.keep_owned(
        [("a0", {}), ("a1", {}), ("b0", {})],
        [("a0", "a1", "related_to", {}), ("a1", "b0", "related_to", {})],
        owned,
    )
    assert _ids(nodes) == {"a0", "a1"}
    assert _keys(edges) == {("a0", "a1", "related_to")}


@pytest.mark.asyncio
async def test_with_access_control_on_a_dataset_reads_its_own_graph(monkeypatch):
    monkeypatch.setattr(subgraph_data, "backend_access_control_enabled", lambda: True)

    assert await get_owned_graph(SharedGraph(), "a") is None
