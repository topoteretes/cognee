"""A dataset's view of a graph shared by every dataset (access control off).

With access control off all datasets write into one graph, so switching to the
dataset's database context changes nothing. Every read for one dataset must
then return only the nodes and edges that dataset owns: an entity another
dataset also mentions is shown, an edge only another dataset drew is not.
"""

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
from cognee.modules.graph import dataset_scope
from cognee.modules.graph.dataset_scope import (
    DatasetScope,
    get_dataset_scope,
    get_shared_graph_scope,
)
from cognee.modules.visualization.graph_stream import stream_graph_events
from cognee.modules.visualization.subgraph_data import fetch_visualization_graph_data

# Dataset "a" owns a0..a3, dataset "b" owns b0..b3, and both mention "shared".
# a0 -> a1 "mentions" was drawn by dataset b only, so a's view must not show it.
NODES = {
    "a0": {"a"},
    "a1": {"a"},
    "a2": {"a"},
    "a3": {"a"},
    "b0": {"b"},
    "b1": {"b"},
    "b2": {"b"},
    "b3": {"b"},
    "shared": {"a", "b"},
}
EDGES = {
    ("shared", "a0", "contains"): {"a"},
    ("a0", "a1", "related_to"): {"a"},
    ("a1", "a2", "related_to"): {"a"},
    ("a2", "a3", "related_to"): {"a"},
    ("a0", "a1", "mentions"): {"b"},
    ("shared", "b0", "contains"): {"b"},
    ("b0", "b1", "related_to"): {"b"},
    ("b1", "b2", "related_to"): {"b"},
    ("b2", "b3", "related_to"): {"b"},
}
A_NODES = {node_id for node_id, owners in NODES.items() if "a" in owners}
A_EDGES = {edge for edge, owners in EDGES.items() if owners == {"a"}}


class SharedGraph:
    """An in-memory graph holding both datasets, with in-graph provenance."""

    def __init__(self):
        self.nodes = [(node_id, {"name": node_id, "type": "Entity"}) for node_id in NODES]
        self.edges = [(source, target, name, {}) for source, target, name in EDGES]

    async def get_graph_metadata(self):
        return {
            GRAPH_DELETE_MODE_KEY: GRAPH_DELETE_MODE_GRAPH_PROVENANCE,
            GRAPH_PROVENANCE_VERSION_KEY: GRAPH_PROVENANCE_VERSION,
        }

    async def find_node_source_refs_by_dataset(self, dataset_id):
        return {node_id: ["ref"] for node_id, owners in NODES.items() if dataset_id in owners}

    async def find_edge_source_refs_by_dataset(self, dataset_id):
        return {
            EdgeIdentity(source, target, name): ["ref"]
            for (source, target, name), owners in EDGES.items()
            if dataset_id in owners
        }

    async def get_graph_data(self):
        return self.nodes, self.edges

    async def get_neighborhood(self, node_ids, depth=1, edge_types=None):
        distance = hop_distances(self.edges, node_ids)
        reached = {node_id for node_id, hops in distance.items() if hops <= depth}
        nodes = [node for node in self.nodes if node[0] in reached]
        edges = [edge for edge in self.edges if edge[0] in reached and edge[1] in reached]
        return nodes, edges

    def iter_bounded_neighborhood(self, *args, **kwargs):
        return GraphDBInterface.iter_bounded_neighborhood(self, *args, **kwargs)

    async def get_entity_type_names(self, entity_ids):
        return {}


def _ids(nodes):
    return {node_id for node_id, _ in nodes}


def _keys(edges):
    return {(source, target, name) for source, target, name, _ in edges}


@pytest.mark.asyncio
async def test_a_bounded_read_returns_only_what_the_dataset_owns():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "a")

    nodes, edges = await fetch_visualization_graph_data(graph, scope=scope)

    assert _ids(nodes) == A_NODES
    assert _keys(edges) == A_EDGES


@pytest.mark.asyncio
async def test_a_full_read_returns_only_what_the_dataset_owns():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "a")

    nodes, edges = await fetch_visualization_graph_data(graph, scope=scope, full=True)

    assert _ids(nodes) == A_NODES
    assert _keys(edges) == A_EDGES


@pytest.mark.asyncio
async def test_seeds_of_another_dataset_do_not_bring_its_nodes_in():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "a")

    nodes, _ = await fetch_visualization_graph_data(graph, scope=scope, seed_node_ids=["b2"])

    assert _ids(nodes) == A_NODES


@pytest.mark.asyncio
async def test_the_node_budget_goes_to_the_datasets_own_nodes():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "a")

    nodes, _ = await fetch_visualization_graph_data(
        graph, scope=scope, seed_node_ids=["a2"], max_nodes=3
    )

    assert len(nodes) == 3
    assert _ids(nodes) <= A_NODES
    assert nodes[0][0] == "a2"


@pytest.mark.asyncio
async def test_a_dataset_that_owns_nothing_gets_an_empty_graph():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "nobody")

    assert await fetch_visualization_graph_data(graph, scope=scope) == ([], [])


@pytest.mark.asyncio
async def test_a_streamed_read_sends_only_what_the_dataset_owns():
    graph = SharedGraph()
    scope = await get_dataset_scope(graph, "a")

    events = [
        event
        async for event in stream_graph_events(
            graph,
            scope=scope,
            query=None,
            seed_node_ids=["b0", "a0"],
            neighborhood_depth=2,
            seed_top_k=10,
            max_nodes=100,
            chunk_size=2,
        )
    ]

    meta = next(data for name, data in events if name == "meta")
    chunks = [data for name, data in events if name == "chunk"]
    assert meta["seeds"] == ["a0"]
    assert {node["id"] for chunk in chunks for node in chunk["nodes"]} == A_NODES
    sent_links = {
        (link["source"], link["target"], link["relation"])
        for chunk in chunks
        for link in chunk["links"]
    }
    assert sent_links == A_EDGES
    assert events[-1] == ("done", {"nodes": len(A_NODES), "links": len(A_EDGES), "chunks": 3})


def test_an_unknown_edge_owner_keeps_edges_by_their_endpoints():
    scope = DatasetScope(node_ids=frozenset({"x", "y"}), edge_keys=None)

    nodes, edges = scope.keep(
        [("x", {}), ("y", {}), ("z", {})],
        [("x", "y", "related_to", {}), ("y", "z", "related_to", {})],
    )

    assert _ids(nodes) == {"x", "y"}
    assert _keys(edges) == {("x", "y", "related_to")}


@pytest.mark.asyncio
async def test_a_ledger_graph_is_scoped_by_its_ledger_rows(monkeypatch):
    graph = MagicMock()

    async def unmarked():
        return {}

    graph.get_graph_metadata = unmarked

    session = MagicMock()

    async def scalars(statement):
        assert "nodes.dataset_id" in str(statement)
        return ["x", "y"]

    session.scalars = scalars

    @asynccontextmanager
    async def get_async_session():
        yield session

    monkeypatch.setattr(
        dataset_scope,
        "get_relational_engine",
        lambda: MagicMock(get_async_session=get_async_session),
    )

    scope = await get_dataset_scope(graph, "a")

    assert scope == DatasetScope(node_ids=frozenset({"x", "y"}), edge_keys=None)


@pytest.mark.asyncio
async def test_with_access_control_on_a_dataset_reads_its_own_graph(monkeypatch):
    monkeypatch.setattr(dataset_scope, "backend_access_control_enabled", lambda: True)

    assert await get_shared_graph_scope(SharedGraph(), "a") is None


@pytest.mark.asyncio
async def test_with_access_control_off_a_dataset_is_scoped(monkeypatch):
    monkeypatch.setattr(dataset_scope, "backend_access_control_enabled", lambda: False)

    scope = await get_shared_graph_scope(SharedGraph(), "a")

    assert scope.node_ids == frozenset(A_NODES)
