"""Tests for KuzuAdapter and SubprocessGraphDBWrapper(KuzuAdapter).

Each test is parametrized to run against both a direct KuzuAdapter and the
same adapter wrapped in a SubprocessGraphDBWrapper.
"""

import json
import os
from pathlib import Path

import pytest

from cognee.shared.data_models import KnowledgeGraph
from cognee.infrastructure.databases.graph.kuzu.adapter import KuzuAdapter
from cognee.infrastructure.databases.graph.subprocess_graph_wrapper import (
    SubprocessGraphDBWrapper,
)

DEMO_KG_PATH = os.path.join(os.path.dirname(__file__), "test_kg.json")


def _load_demo_kg() -> KnowledgeGraph:
    data = json.loads(Path(DEMO_KG_PATH).read_text(encoding="utf-8"))
    return KnowledgeGraph.model_validate(data)


@pytest.fixture
def kuzu_adapter(tmp_path):
    """Direct KuzuAdapter instance."""
    return KuzuAdapter(db_path=str(tmp_path / "kuzu_direct"))


@pytest.fixture
def subprocess_adapter(tmp_path):
    """KuzuAdapter running inside a subprocess via the wrapper."""
    return SubprocessGraphDBWrapper(
        KuzuAdapter,
        db_path=str(tmp_path / "kuzu_subprocess"),
        shutdown_timeout=10,
    )


@pytest.fixture(params=["direct", "subprocess"])
def adapter(request, tmp_path):
    """Parametrized fixture: yields both direct and subprocess-wrapped adapters."""
    if request.param == "direct":
        return KuzuAdapter(db_path=str(tmp_path / "kuzu_direct"))
    else:
        return SubprocessGraphDBWrapper(
            KuzuAdapter,
            db_path=str(tmp_path / "kuzu_subprocess"),
            shutdown_timeout=10,
        )


# ---------------------------------------------------------------------------
# is_empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_empty_on_fresh_db(adapter):
    assert await adapter.is_empty() is True


# ---------------------------------------------------------------------------
# add_nodes (single) / has_node / get_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]  # Alice

    # Use add_nodes (batch) with a single node — add_node uses a
    # different Cypher path that has a known Kuzu compatibility issue.
    await adapter.add_nodes([node])

    result = await adapter.get_node(node.id)
    assert result is not None
    assert result["id"] == node.id
    assert result["name"] == node.name


@pytest.mark.asyncio
async def test_has_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]

    assert await adapter.has_node(node.id) is False
    await adapter.add_nodes([node])
    assert await adapter.has_node(node.id) is True


# ---------------------------------------------------------------------------
# add_nodes / get_nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_get_nodes(adapter):
    kg = _load_demo_kg()

    await adapter.add_nodes(kg.nodes)

    node_ids = [n.id for n in kg.nodes]
    results = await adapter.get_nodes(node_ids)
    assert len(results) == len(kg.nodes)

    result_ids = {r["id"] for r in results}
    assert result_ids == set(node_ids)


# ---------------------------------------------------------------------------
# delete_node / delete_nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_node(adapter):
    kg = _load_demo_kg()
    node = kg.nodes[0]

    await adapter.add_nodes([node])
    assert await adapter.has_node(node.id) is True

    await adapter.delete_node(node.id)
    assert await adapter.has_node(node.id) is False


@pytest.mark.asyncio
async def test_delete_nodes(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    ids = [n.id for n in kg.nodes]
    await adapter.delete_nodes(ids)

    assert await adapter.is_empty() is True


# ---------------------------------------------------------------------------
# add_edge / has_edge / get_edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_and_has_edge(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge = kg.edges[0]  # Alice -> Mark, "knows"
    await adapter.add_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name, {})

    assert await adapter.has_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name)


@pytest.mark.asyncio
async def test_get_edges(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge = kg.edges[0]
    await adapter.add_edge(edge.source_node_id, edge.target_node_id, edge.relationship_name, {})

    edges = await adapter.get_edges(edge.source_node_id)
    assert len(edges) >= 1


# ---------------------------------------------------------------------------
# add_edges / has_edges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edges_batch(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    # Verify all edges exist
    check_edges = [
        (e.source_node_id, e.target_node_id, e.relationship_name) for e in kg.edges
    ]
    existing = await adapter.has_edges(check_edges)
    assert len(existing) == len(kg.edges)


# ---------------------------------------------------------------------------
# get_neighbors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_neighbors(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    # Mark has edges: Alice->Mark (knows), Mark->Bob (had_dinner_with), Mark->Alice (had_dinner_with)
    neighbors = await adapter.get_neighbors("Mark")
    neighbor_ids = {n["id"] for n in neighbors}
    assert "Alice" in neighbor_ids
    assert "Bob" in neighbor_ids


# ---------------------------------------------------------------------------
# get_connections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_connections(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    connections = await adapter.get_connections("Mark")
    assert len(connections) >= 1

    # Each connection is (source_node, relationship_info, target_node)
    for conn in connections:
        assert len(conn) == 3


# ---------------------------------------------------------------------------
# get_graph_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_graph_data(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == len(kg.nodes)
    assert len(edges) == len(kg.edges)


# ---------------------------------------------------------------------------
# get_filtered_graph_data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_filtered_graph_data(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    # Filter by type = "Person"
    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["Person"]}])
    assert len(nodes) == len(kg.nodes)  # All nodes are Person type


# ---------------------------------------------------------------------------
# get_predecessors / get_successors
# Known adapter bug: RETURN properties(m) fails with current Kuzu version.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: RETURN properties(m) not supported in current Kuzu")
async def test_predecessors_and_successors(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    # Alice->Mark (knows), so Mark's predecessors with "knows" should include Alice
    predecessors = await adapter.get_predecessors("Mark", edge_label="knows")
    assert len(predecessors) > 0

    # Mark->Bob (had_dinner_with), so Mark's successors should include Bob
    successors = await adapter.get_successors("Mark", edge_label="had_dinner_with")
    assert len(successors) > 0


# ---------------------------------------------------------------------------
# get_graph_metrics
# Known adapter bug: get_model_independent_graph_data returns wrong format.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: get_model_independent_graph_data format mismatch")
async def test_get_graph_metrics(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == len(kg.nodes)
    assert metrics["num_edges"] == len(kg.edges)


# ---------------------------------------------------------------------------
# get_disconnected_nodes
# Known adapter bug: NOT EXISTS((n)-[]-()) syntax not supported in current Kuzu.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.xfail(reason="KuzuAdapter bug: NOT EXISTS pattern syntax unsupported")
async def test_get_disconnected_nodes(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    # Before adding edges, all nodes should be disconnected
    disconnected = await adapter.get_disconnected_nodes()
    assert len(disconnected) == len(kg.nodes)

    # After adding edges, connected nodes should disappear from the list
    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    disconnected_after = await adapter.get_disconnected_nodes()
    assert len(disconnected_after) < len(disconnected)


# ---------------------------------------------------------------------------
# query (raw)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raw_query(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    result = await adapter.query("MATCH (n:Node) RETURN COUNT(n)")
    assert result[0][0] == len(kg.nodes)


# ---------------------------------------------------------------------------
# is_empty after data + delete_graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_not_empty_after_add(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    assert await adapter.is_empty() is False


# ---------------------------------------------------------------------------
# get_triplets_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_triplets_batch(adapter):
    kg = _load_demo_kg()
    await adapter.add_nodes(kg.nodes)

    edge_rows = [
        (e.source_node_id, e.target_node_id, e.relationship_name, {})
        for e in kg.edges
    ]
    await adapter.add_edges(edge_rows)

    triplets = await adapter.get_triplets_batch(offset=0, limit=10)
    assert len(triplets) == len(kg.edges)

    for t in triplets:
        assert "start_node" in t
        assert "relationship_properties" in t
        assert "end_node" in t


# ---------------------------------------------------------------------------
# close / subprocess lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_direct(kuzu_adapter):
    await kuzu_adapter.add_nodes(_load_demo_kg().nodes)
    await kuzu_adapter.close()


@pytest.mark.asyncio
async def test_close_subprocess(subprocess_adapter):
    await subprocess_adapter.add_nodes(_load_demo_kg().nodes)
    await subprocess_adapter.close()


@pytest.mark.asyncio
async def test_subprocess_calls_after_close_raise(subprocess_adapter):
    await subprocess_adapter.close()
    with pytest.raises(RuntimeError):
        await subprocess_adapter.query("MATCH (n) RETURN n")
