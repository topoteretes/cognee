"""Unit tests for TursoAdapter using an in-memory SQLite database."""

import json
import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter


@pytest_asyncio.fixture
async def adapter():
    """In-memory SQLite adapter — fast, no disk I/O, isolated per test."""
    a = TursoAdapter(connection_string="sqlite+aiosqlite:///:memory:")
    await a.initialize()
    yield a
    await a.delete_graph()
    await a.close()


@pytest.mark.asyncio
async def test_is_empty_on_fresh_db(adapter):
    assert await adapter.is_empty()


@pytest.mark.asyncio
async def test_add_and_get_node(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    node = await adapter.get_node("n1")
    assert node is not None
    assert node["name"] == "Alice"
    assert node["type"] == "Person"


@pytest.mark.asyncio
async def test_add_nodes_bulk(adapter):
    nodes = [
        ("n1", {"name": "Alice", "type": "Person"}),
        ("n2", {"name": "Bob", "type": "Person"}),
        ("n3", {"name": "Carol", "type": "Person"}),
    ]
    await adapter.add_nodes(nodes)
    results = await adapter.get_nodes(["n1", "n2", "n3"])
    assert len(results) == 3


@pytest.mark.asyncio
async def test_upsert_node(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n1", {"name": "Alice Updated", "type": "Person"})
    node = await adapter.get_node("n1")
    assert node["name"] == "Alice Updated"


@pytest.mark.asyncio
async def test_delete_node(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.delete_node("n1")
    assert await adapter.get_node("n1") is None


@pytest.mark.asyncio
async def test_add_and_has_edge(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")
    assert await adapter.has_edge("n1", "n2", "KNOWS")
    assert not await adapter.has_edge("n2", "n1", "KNOWS")


@pytest.mark.asyncio
async def test_has_edges_returns_found_tuples(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    found = await adapter.has_edges([
        ("n1", "n2", "KNOWS"),
        ("n2", "n1", "KNOWS"),   # does not exist
    ])
    assert len(found) == 1
    assert found[0] == ("n1", "n2", "KNOWS")


@pytest.mark.asyncio
async def test_get_edges_returns_src_rel_tgt_tuples(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    edges = await adapter.get_edges("n1")
    assert len(edges) == 1
    src, rel, tgt = edges[0]
    assert isinstance(src, dict) and src["id"] == "n1"
    assert rel == "KNOWS"
    assert isinstance(tgt, dict) and tgt["id"] == "n2"


@pytest.mark.asyncio
async def test_get_neighbors(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_node("n3", {"name": "Carol", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")
    await adapter.add_edge("n1", "n3", "KNOWS")

    neighbors = await adapter.get_neighbors("n1")
    ids = {n["id"] for n in neighbors}
    assert ids == {"n2", "n3"}


@pytest.mark.asyncio
async def test_get_connections(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS", {"since": 2020})

    conns = await adapter.get_connections("n1")
    assert len(conns) == 1
    src, edge, tgt = conns[0]
    assert src["id"] == "n1"
    assert edge["relationship_name"] == "KNOWS"
    assert tgt["id"] == "n2"


@pytest.mark.asyncio
async def test_get_graph_data_format(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    nodes, edges = await adapter.get_graph_data()

    # nodes must be (id, props_dict) tuples
    assert all(isinstance(n, tuple) and len(n) == 2 for n in nodes)
    node_ids = {n[0] for n in nodes}
    assert node_ids == {"n1", "n2"}

    # edges must be (src, tgt, rel, props) tuples
    assert all(isinstance(e, tuple) and len(e) == 4 for e in edges)
    assert edges[0][:3] == ("n1", "n2", "KNOWS")


@pytest.mark.asyncio
async def test_get_graph_metrics(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == 2
    assert metrics["num_edges"] == 1
    assert "mean_degree" in metrics
    assert "edge_density" in metrics
    assert "num_connected_components" in metrics


@pytest.mark.asyncio
async def test_delete_graph(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")
    await adapter.delete_graph()
    assert await adapter.is_empty()


@pytest.mark.asyncio
async def test_get_neighborhood(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_node("n3", {"name": "Carol", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")
    await adapter.add_edge("n2", "n3", "KNOWS")

    nodes, edges = await adapter.get_neighborhood(["n1"], depth=1)
    node_ids = {n[0] for n in nodes}
    # depth=1: n1 and its direct neighbor n2
    assert "n1" in node_ids
    assert "n2" in node_ids
    assert "n3" not in node_ids


@pytest.mark.asyncio
async def test_get_triplets_batch(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    triplets = await adapter.get_triplets_batch(offset=0, limit=10)
    assert len(triplets) == 1
    assert triplets[0]["start_node"]["id"] == "n1"
    assert triplets[0]["relationship_properties"]["relationship_name"] == "KNOWS"
    assert triplets[0]["end_node"]["id"] == "n2"


@pytest.mark.asyncio
async def test_cascade_delete_edges_on_node_delete(adapter):
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    await adapter.delete_node("n1")
    # Edge should be cascade-deleted
    edges = await adapter.get_edges("n2")
    assert len(edges) == 0


@pytest.mark.asyncio
async def test_query_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        await adapter.query("MATCH (n) RETURN n")
