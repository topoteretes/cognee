"""Unit tests for the Postgres graph adapter.

Requires a running Postgres instance. Connection defaults:
  DB_HOST=localhost, DB_PORT=5432, DB_USERNAME=cognee,
  DB_PASSWORD=cognee, DB_NAME=cognee_db

Override via environment variables if needed.
"""

import os
import json
import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter


# -- Session-scoped event loop so the async engine's connection pool
#    stays on a single loop across all tests.


@pytest.fixture(scope="session")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -- Fixture: real Postgres --


@pytest_asyncio.fixture
async def adapter():
    """Create adapter backed by a real Postgres database.

    Initializes schema, yields the adapter, then cleans up all graph
    tables so tests are isolated.
    """
    username = os.environ.get("DB_USERNAME", "cognee")
    password = os.environ.get("DB_PASSWORD", "cognee")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    database = os.environ.get("DB_NAME", "cognee_db")
    connection_string = f"postgresql+asyncpg://{username}:{password}@{host}:{port}/{database}"
    a = PostgresAdapter(connection_string=connection_string)

    # Create tables and indexes
    await a.initialize()

    # Clean slate before each test
    await a.delete_graph()

    yield a

    # Clean up after each test
    await a.delete_graph()


# -- Helpers --


class _FakeDataPoint:
    """Minimal DataPoint-like object for testing."""

    def __init__(self, id, name="", type="", **extra):
        self._data = {"id": str(id), "name": name, "type": type, **extra}

    def model_dump(self):
        return dict(self._data)


# -- Tests: node operations --


@pytest.mark.asyncio
async def test_is_empty_on_fresh_db(adapter):
    assert await adapter.is_empty() is True


@pytest.mark.asyncio
async def test_add_and_get_node(adapter):
    node = _FakeDataPoint(id="n1", name="Alice", type="Person", age=30)
    await adapter.add_node(node)

    result = await adapter.get_node("n1")
    assert result is not None
    assert result["id"] == "n1"
    assert result["name"] == "Alice"
    assert result["type"] == "Person"


@pytest.mark.asyncio
async def test_add_node_string_form(adapter):
    await adapter.add_node("n2", properties={"name": "Bob", "type": "Person"})
    result = await adapter.get_node("n2")
    assert result is not None
    assert result["name"] == "Bob"


@pytest.mark.asyncio
async def test_add_nodes_batch(adapter):
    nodes = [
        _FakeDataPoint(id="a", name="A", type="X"),
        _FakeDataPoint(id="b", name="B", type="Y"),
        _FakeDataPoint(id="c", name="C", type="X"),
    ]
    await adapter.add_nodes(nodes)

    results = await adapter.get_nodes(["a", "b", "c"])
    assert len(results) == 3
    names = {r["name"] for r in results}
    assert names == {"A", "B", "C"}


@pytest.mark.asyncio
async def test_add_nodes_upsert(adapter):
    """add_nodes should update existing nodes, not fail."""
    await adapter.add_nodes([_FakeDataPoint(id="u1", name="V1", type="T")])
    await adapter.add_nodes([_FakeDataPoint(id="u1", name="V2", type="T")])

    result = await adapter.get_node("u1")
    assert result["name"] == "V2"


@pytest.mark.asyncio
async def test_is_empty_after_add(adapter):
    await adapter.add_nodes([_FakeDataPoint(id="x", name="X", type="T")])
    assert await adapter.is_empty() is False


@pytest.mark.asyncio
async def test_delete_node(adapter):
    await adapter.add_nodes([_FakeDataPoint(id="d1", name="D", type="T")])
    await adapter.delete_node("d1")
    assert await adapter.get_node("d1") is None


@pytest.mark.asyncio
async def test_delete_nodes_batch(adapter):
    nodes = [_FakeDataPoint(id=f"dn{i}", name=f"N{i}", type="T") for i in range(3)]
    await adapter.add_nodes(nodes)
    await adapter.delete_nodes(["dn0", "dn1", "dn2"])
    results = await adapter.get_nodes(["dn0", "dn1", "dn2"])
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_nodes_empty_ids(adapter):
    assert await adapter.get_nodes([]) == []


# -- Tests: edge operations --


@pytest.mark.asyncio
async def test_add_and_has_edge(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="e1", name="A", type="T"),
            _FakeDataPoint(id="e2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("e1", "e2", "KNOWS", {"since": 2020})

    assert await adapter.has_edge("e1", "e2", "KNOWS") is True
    assert await adapter.has_edge("e1", "e2", "LIKES") is False
    assert await adapter.has_edge("e2", "e1", "KNOWS") is False


@pytest.mark.asyncio
async def test_add_edges_batch(adapter):
    nodes = [_FakeDataPoint(id=f"be{i}", name=f"N{i}", type="T") for i in range(3)]
    await adapter.add_nodes(nodes)

    edges = [
        ("be0", "be1", "R1", {"w": 1}),
        ("be1", "be2", "R2", {"w": 2}),
        ("be0", "be2", "R3", {}),
    ]
    await adapter.add_edges(edges)

    existing = await adapter.has_edges(
        [
            ("be0", "be1", "R1"),
            ("be1", "be2", "R2"),
            ("be0", "be2", "R3"),
            ("be2", "be0", "R1"),  # does not exist
        ]
    )
    assert len(existing) == 3


@pytest.mark.asyncio
async def test_add_edges_upsert(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="eu1", name="A", type="T"),
            _FakeDataPoint(id="eu2", name="B", type="T"),
        ]
    )
    await adapter.add_edges([("eu1", "eu2", "R", {"v": 1})])
    await adapter.add_edges([("eu1", "eu2", "R", {"v": 2})])

    # Should not fail, and edge should be updated (not duplicated)
    existing = await adapter.has_edges([("eu1", "eu2", "R")])
    assert len(existing) == 1


@pytest.mark.asyncio
async def test_get_edges(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="ge1", name="A", type="T"),
            _FakeDataPoint(id="ge2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("ge1", "ge2", "LINKS")

    edges = await adapter.get_edges("ge1")
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_cascade_delete(adapter):
    """Deleting a node should cascade-delete its edges."""
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="cd1", name="A", type="T"),
            _FakeDataPoint(id="cd2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("cd1", "cd2", "R")
    await adapter.delete_node("cd1")

    assert await adapter.has_edge("cd1", "cd2", "R") is False


# -- Tests: neighbor and connection queries --


@pytest.mark.asyncio
async def test_get_neighbors(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="nb1", name="Center", type="T"),
            _FakeDataPoint(id="nb2", name="Left", type="T"),
            _FakeDataPoint(id="nb3", name="Right", type="T"),
        ]
    )
    await adapter.add_edges(
        [
            ("nb1", "nb2", "R1", {}),
            ("nb3", "nb1", "R2", {}),
        ]
    )

    neighbors = await adapter.get_neighbors("nb1")
    neighbor_ids = {n["id"] for n in neighbors}
    assert neighbor_ids == {"nb2", "nb3"}


@pytest.mark.asyncio
async def test_get_connections(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="cn1", name="A", type="T"),
            _FakeDataPoint(id="cn2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("cn1", "cn2", "LINKED")

    connections = await adapter.get_connections("cn1")
    assert len(connections) == 1
    src, edge, tgt = connections[0]
    assert edge["relationship_name"] == "LINKED"


# -- Tests: graph-wide reads --


@pytest.mark.asyncio
async def test_get_graph_data(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="gd1", name="A", type="T"),
            _FakeDataPoint(id="gd2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("gd1", "gd2", "R")

    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 2
    assert len(edges) == 1
    assert edges[0][2] == "R"


@pytest.mark.asyncio
async def test_get_graph_data_empty(adapter):
    nodes, edges = await adapter.get_graph_data()
    assert nodes == []
    assert edges == []


@pytest.mark.asyncio
async def test_get_nodeset_subgraph(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="ns1", name="Alpha", type="Entity"),
            _FakeDataPoint(id="ns2", name="Beta", type="Entity"),
            _FakeDataPoint(id="ns3", name="Gamma", type="Other"),
        ]
    )
    await adapter.add_edges(
        [
            ("ns1", "ns2", "R", {}),
            ("ns1", "ns3", "R", {}),
        ]
    )

    # Create a fake type with __name__ == "Entity"
    class Entity:
        pass

    nodes, edges = await adapter.get_nodeset_subgraph(Entity, ["Alpha"])
    node_ids = {n[0] for n in nodes}

    # Should include ns1 (primary) plus neighbors ns2 and ns3
    assert "ns1" in node_ids
    assert len(node_ids) >= 2


@pytest.mark.asyncio
async def test_get_filtered_graph_data(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="fg1", name="A", type="X"),
            _FakeDataPoint(id="fg2", name="B", type="Y"),
            _FakeDataPoint(id="fg3", name="C", type="X"),
        ]
    )
    await adapter.add_edge("fg1", "fg3", "R")

    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["X"]}])
    node_ids = {n[0] for n in nodes}
    assert node_ids == {"fg1", "fg3"}
    assert len(edges) == 1


# -- Tests: metrics --


@pytest.mark.asyncio
async def test_get_graph_metrics_basic(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="m1", name="A", type="T"),
            _FakeDataPoint(id="m2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("m1", "m2", "R")

    metrics = await adapter.get_graph_metrics()
    assert metrics["num_nodes"] == 2
    assert metrics["num_edges"] == 1


# -- Tests: delete_graph --


@pytest.mark.asyncio
async def test_delete_graph(adapter):
    await adapter.add_nodes([_FakeDataPoint(id="dg1", name="A", type="T")])
    await adapter.add_nodes([_FakeDataPoint(id="dg2", name="B", type="T")])
    await adapter.add_edge("dg1", "dg2", "R")

    await adapter.delete_graph()
    assert await adapter.is_empty() is True


# -- Tests: triplets --


@pytest.mark.asyncio
async def test_get_triplets_batch(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="t1", name="Start", type="T"),
            _FakeDataPoint(id="t2", name="End", type="T"),
        ]
    )
    await adapter.add_edge("t1", "t2", "CONNECTS")

    triplets = await adapter.get_triplets_batch(offset=0, limit=10)
    assert len(triplets) == 1
    assert triplets[0]["start_node"]["name"] == "Start"
    assert triplets[0]["end_node"]["name"] == "End"
    assert triplets[0]["relationship_properties"]["relationship_name"] == "CONNECTS"


@pytest.mark.asyncio
async def test_get_triplets_batch_offset(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="to1", name="A", type="T"),
            _FakeDataPoint(id="to2", name="B", type="T"),
            _FakeDataPoint(id="to3", name="C", type="T"),
        ]
    )
    await adapter.add_edges(
        [
            ("to1", "to2", "R1", {}),
            ("to1", "to3", "R2", {}),
        ]
    )

    all_triplets = await adapter.get_triplets_batch(offset=0, limit=10)
    assert len(all_triplets) == 2

    one_triplet = await adapter.get_triplets_batch(offset=1, limit=1)
    assert len(one_triplet) == 1


@pytest.mark.asyncio
async def test_get_triplets_batch_validation(adapter):
    with pytest.raises(ValueError):
        await adapter.get_triplets_batch(offset=-1, limit=10)
    with pytest.raises(ValueError):
        await adapter.get_triplets_batch(offset=0, limit=-1)


# -- Tests: query raises --


@pytest.mark.asyncio
async def test_query_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        await adapter.query("MATCH (n) RETURN n")
