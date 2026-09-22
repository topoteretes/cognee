"""Tests for the Turso graph adapter against a real Turso (rewrite engine) database file."""

import json

import pytest
import pytest_asyncio
from sqlalchemy import text

pytest.importorskip("turso", reason="pyturso not installed")

from cognee.infrastructure.databases.graph.turso.adapter import TursoAdapter
from cognee.infrastructure.databases.turso import turso_url


@pytest_asyncio.fixture
async def adapter(tmp_path):
    """File-backed adapter on the rewrite engine — isolated per test."""
    a = TursoAdapter(database_path=str(tmp_path / "graph.db"))
    await a.initialize()
    yield a
    await a.delete_graph()
    await a.close()


@pytest.mark.asyncio
async def test_runs_on_the_turso_rewrite_engine(adapter):
    """``turso_version()`` exists only on the rewrite engine; SQLite has no such function."""
    assert adapter.engine.dialect.driver == "cognee_turso"
    async with adapter.engine.connect() as connection:
        version = (await connection.execute(text("SELECT turso_version()"))).scalar()
    assert version


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

    found = await adapter.has_edges(
        [
            ("n1", "n2", "KNOWS"),
            ("n2", "n1", "KNOWS"),  # does not exist
        ]
    )
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

    nodes, _edges = await adapter.get_neighborhood(["n1"], depth=1)
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


# --- Regression tests for the maintainer rework (SDK-176) -----------------------


@pytest.mark.asyncio
async def test_add_nodes_and_edges_accept_provenance_kwargs(adapter):
    """The storage path always passes source_ref_key/pipeline_run_id; the adapter
    must accept (and ignore) them instead of raising TypeError at cognify time."""
    await adapter.add_nodes(
        [("n1", {"name": "Alice", "type": "Person"})],
        source_ref_key="ref",
        pipeline_run_id="run",
    )
    await adapter.add_edges(
        [("n1", "n1", "SELF", {})],
        source_ref_key="ref",
        pipeline_run_id="run",
    )
    assert await adapter.get_node("n1") is not None


@pytest.mark.asyncio
async def test_upsert_node_preserves_edges(adapter):
    """Re-adding an existing node (upsert) must NOT delete its edges. Regression
    for INSERT OR REPLACE, which cascade-deleted edges via ON DELETE CASCADE."""
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    await adapter.add_node("n2", {"name": "Bob", "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")

    # Re-add n1 with changed properties (happens on every re-cognify / shared entity).
    await adapter.add_node("n1", {"name": "Alice Updated", "type": "Person"})

    assert (await adapter.get_node("n1"))["name"] == "Alice Updated"
    assert len(await adapter.get_edges("n1")) == 1


@pytest.mark.asyncio
async def test_remove_belongs_to_set_tags(adapter):
    await adapter.add_node(
        "n1", {"name": "Alice", "type": "Person", "belongs_to_set": ["set_a", "set_b"]}
    )
    await adapter.add_node("n2", {"name": "Bob", "type": "Person", "belongs_to_set": ["set_c"]})

    await adapter.remove_belongs_to_set_tags(["set_a"])

    assert (await adapter.get_node("n1"))["belongs_to_set"] == ["set_b"]
    assert (await adapter.get_node("n2"))["belongs_to_set"] == ["set_c"]


@pytest.mark.asyncio
async def test_get_filtered_graph_data_empty_values_matches_nothing(adapter):
    """An empty filter value list matches nothing without an "IN ()" syntax error."""
    await adapter.add_node("n1", {"name": "Alice", "type": "Person"})
    nodes, edges = await adapter.get_filtered_graph_data([{"type": []}])
    assert nodes == [] and edges == []


@pytest.mark.asyncio
async def test_get_nodeset_subgraph_empty_names(adapter):
    class Person:
        pass

    nodes, edges = await adapter.get_nodeset_subgraph(Person, [])
    assert nodes == [] and edges == []


@pytest.mark.asyncio
async def test_bulk_add_delete_large_id_set(adapter):
    """Bulk add/get/delete over a large id set must not raise "too many SQL
    variables" — id lists are bound as a single JSON array via json_each."""
    ids = [f"n{i}" for i in range(2000)]
    await adapter.add_nodes([(i, {"name": i, "type": "Person"}) for i in ids])
    assert len(await adapter.get_nodes(ids)) == 2000
    await adapter.delete_nodes(ids)
    assert await adapter.is_empty()


@pytest.mark.asyncio
async def test_get_neighborhood_depth_and_edge_types(adapter):
    for n in ("n1", "n2", "n3", "n4"):
        await adapter.add_node(n, {"name": n, "type": "Person"})
    await adapter.add_edge("n1", "n2", "KNOWS")
    await adapter.add_edge("n2", "n3", "KNOWS")
    await adapter.add_edge("n3", "n4", "LIKES")

    nodes, _ = await adapter.get_neighborhood(["n1"], depth=2)
    assert {n[0] for n in nodes} == {"n1", "n2", "n3"}

    # edge_types restricts traversal to KNOWS edges only.
    nodes, _ = await adapter.get_neighborhood(["n1"], depth=3, edge_types=["KNOWS"])
    assert "n4" not in {n[0] for n in nodes}


@pytest.mark.asyncio
async def test_factory_returns_turso_adapter_and_rejects_remote(tmp_path):
    from cognee.infrastructure.databases.graph.get_graph_engine import (
        create_graph_engine,
        graph_engine_cache,
    )

    db_path = str(tmp_path / "graph.db")
    kwargs = {
        "graph_database_provider": "turso",
        "graph_file_path": "",
        "graph_database_url": db_path,
        "graph_database_key": "",
    }
    engine = create_graph_engine(**kwargs)
    try:
        assert isinstance(engine, TursoAdapter)
        assert engine.db_uri == turso_url(db_path)
        assert engine.database_path == db_path
    finally:
        graph_engine_cache.evict(**kwargs)

    # A set auth token (remote) is rejected: remote Turso is not supported in this version.
    with pytest.raises(OSError):
        create_graph_engine(
            graph_database_provider="turso",
            graph_file_path="",
            graph_database_url=db_path,
            graph_database_key="token",
        )


@pytest.mark.asyncio
async def test_get_graph_metrics_counts_components_with_isolated_nodes(adapter):
    await adapter.add_nodes(
        [
            ("a", {"name": "A", "type": "T"}),
            ("b", {"name": "B", "type": "T"}),
            ("c", {"name": "C", "type": "T"}),
            ("lonely", {"name": "L", "type": "T"}),
        ]
    )
    await adapter.add_edges([("a", "b", "KNOWS", {}), ("b", "c", "KNOWS", {})])

    metrics = await adapter.get_graph_metrics()

    assert metrics["num_nodes"] == 4
    assert metrics["num_edges"] == 2
    assert metrics["num_connected_components"] == 2
    assert metrics["sizes_of_connected_components"] == [3, 1]


@pytest.mark.asyncio
async def test_get_neighborhood_is_bounded_by_depth(adapter):
    ids = [f"n{i}" for i in range(5)]
    await adapter.add_nodes([(i, {"name": i, "type": "T"}) for i in ids])
    await adapter.add_edges([(ids[i], ids[i + 1], "NEXT", {}) for i in range(4)])

    for depth in range(4):
        nodes, edges = await adapter.get_neighborhood(["n0"], depth=depth)
        assert {n[0] for n in nodes} == set(ids[: depth + 1])
        assert len(edges) == depth


@pytest.mark.asyncio
async def test_get_nodeset_subgraph_with_data(adapter):
    node_set_type = type("NodeSet", (), {})
    await adapter.add_nodes(
        [
            ("ns", {"name": "Set", "type": "NodeSet"}),
            ("x", {"name": "X", "type": "Entity", "belongs_to_set": ["Set"]}),
            ("y", {"name": "Y", "type": "Entity"}),
        ]
    )
    await adapter.add_edges([("x", "ns", "belongs_to_set", {}), ("x", "y", "RELATES", {})])

    nodes, edges = await adapter.get_nodeset_subgraph(node_type=node_set_type, node_name=["Set"])

    node_ids = {n[0] for n in nodes}
    assert {"ns", "x"} <= node_ids
    assert ("x", "ns", "belongs_to_set") in {(e[0], e[1], e[2]) for e in edges}


@pytest.mark.asyncio
async def test_data_persists_across_reopen(tmp_path):
    path = str(tmp_path / "persist.db")
    first = TursoAdapter(database_path=path)
    await first.initialize()
    await first.add_node("keep", {"name": "Keep", "type": "T"})
    await first.close()

    second = TursoAdapter(database_path=path)
    await second.initialize()
    try:
        node = await second.get_node("keep")
        assert node is not None and node["name"] == "Keep"
    finally:
        await second.close()
