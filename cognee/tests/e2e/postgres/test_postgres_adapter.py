"""Unit tests for the Postgres graph adapter.

Requires a running Postgres instance. Connection defaults:
  DB_HOST=localhost, DB_PORT=5432, DB_USERNAME=cognee,
  DB_PASSWORD=cognee, DB_NAME=cognee_db

Override via environment variables if needed.
"""

import asyncio
import os
from copy import deepcopy
from uuid import uuid4

import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.postgres_demo.adapter import PostgresDemoAdapter
from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key


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
    a = PostgresDemoAdapter(connection_string=connection_string)

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
    properties = {"id": "wrong", "name": "Bob", "type": "Person"}
    await adapter.add_node("n2", properties=properties)

    assert properties == {"id": "wrong", "name": "Bob", "type": "Person"}
    result = await adapter.get_node("n2")
    assert result is not None
    assert result["name"] == "Bob"
    assert await adapter.get_node("wrong") is None


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
async def test_duplicate_nodes_and_edges_use_the_last_value(adapter):
    await adapter.add_nodes(
        [
            ("duplicate", {"name": "first", "type": "T"}),
            ("other", {"name": "Other", "type": "T"}),
            ("duplicate", {"name": "last", "type": "T"}),
        ]
    )
    await adapter.add_edges(
        [
            ("duplicate", "other", "R", {"value": "first"}),
            ("duplicate", "other", "R", {"value": "last"}),
        ]
    )

    assert (await adapter.get_node("duplicate"))["name"] == "last"
    connection = (await adapter.get_connections("duplicate"))[0]
    assert connection[1]["value"] == "last"


@pytest.mark.asyncio
async def test_add_nodes_does_not_mutate_caller_properties(adapter):
    """Preparing a write must copy: sanitization rewrites values the caller still owns."""
    properties = {"name": "Ca\0ller", "type": "T", "nested": {"value": "a\0b"}}
    original = deepcopy(properties)

    await adapter.add_nodes([("caller-node", properties)])

    assert properties == original


@pytest.mark.asyncio
async def test_tuple_id_wins_over_conflicting_properties_id(adapter):
    await adapter.add_nodes([("authoritative", {"id": "ignored", "name": "N", "type": "T"})])

    assert (await adapter.get_node("authoritative"))["id"] == "authoritative"
    assert await adapter.get_node("ignored") is None


@pytest.mark.asyncio
async def test_node_and_edge_payloads_are_sanitized(adapter):
    await adapter.add_nodes(
        [
            ("left\0", {"name": "Le\0ft", "type": "T", "nested": {"value": "a\0b"}}),
            ("right", {"name": "Right", "type": "T"}),
        ]
    )
    await adapter.add_edge("left\0", "right", "RE\0L", {"value": "c\0d"})

    node = await adapter.get_node("left")
    connection = (await adapter.get_connections("left"))[0]
    assert node["name"] == "Left"
    assert node["nested"] == {"value": "ab"}
    assert connection[1]["relationship_name"] == "REL"
    assert connection[1]["value"] == "cd"


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
async def test_has_edges_returns_only_present_edges(adapter):
    await adapter.add_nodes([_FakeDataPoint(id=f"he{i}", name=f"N{i}", type="T") for i in range(3)])
    present = [("he0", "he1", "R1"), ("he1", "he2", "R2")]
    await adapter.add_edges([(*edge, {}) for edge in present])

    found = await adapter.has_edges(
        [present[0], ("he0", "he1", "NOPE"), present[1], ("he1", "he0", "R1")]
    )

    assert set(found) == set(present)
    assert await adapter.has_edges([]) == []


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


@pytest.mark.asyncio
async def test_node_write_rolls_back_when_provenance_fails(adapter, monkeypatch):
    async def fail_transition(*args, **kwargs):
        raise RuntimeError("provenance failed")

    monkeypatch.setattr(adapter, "_update_node_provenance", fail_transition)
    source_ref = make_source_ref_key(uuid4(), uuid4())

    with pytest.raises(RuntimeError, match="provenance failed"):
        await adapter.add_nodes(
            [("rollback-node", {"name": "Rollback", "type": "T"})],
            source_ref_key=source_ref,
        )

    assert await adapter.has_node("rollback-node") is False


@pytest.mark.asyncio
async def test_edge_write_rolls_back_when_provenance_fails(adapter, monkeypatch):
    await adapter.add_nodes(
        [
            ("rollback-source", {"name": "Source", "type": "T"}),
            ("rollback-target", {"name": "Target", "type": "T"}),
        ]
    )

    async def fail_transition(*args, **kwargs):
        raise RuntimeError("provenance failed")

    monkeypatch.setattr(adapter, "_update_edge_provenance", fail_transition)
    source_ref = make_source_ref_key(uuid4(), uuid4())

    with pytest.raises(RuntimeError, match="provenance failed"):
        await adapter.add_edges(
            [("rollback-source", "rollback-target", "R", {})],
            source_ref_key=source_ref,
        )

    assert await adapter.has_edge("rollback-source", "rollback-target", "R") is False


@pytest.mark.asyncio
async def test_concurrent_provenance_updates_across_adapter_instances(adapter):
    second_adapter = PostgresDemoAdapter(adapter.db_uri)
    node_id = "concurrent-node"
    edge = EdgeIdentity(node_id, node_id, "LOOP")
    node_keys = [make_source_ref_key(uuid4(), uuid4()) for _ in range(2)]
    edge_keys = [make_source_ref_key(uuid4(), uuid4()) for _ in range(2)]

    try:
        await adapter.add_node(node_id, {"name": "Node", "type": "T"})
        await adapter.add_edge(node_id, node_id, edge.relationship_name)

        await asyncio.wait_for(
            asyncio.gather(
                adapter.attach_node_source_refs([node_id], [node_keys[0]]),
                second_adapter.attach_node_source_refs([node_id], [node_keys[1]]),
                adapter.attach_edge_source_refs([edge], [edge_keys[0]]),
                second_adapter.attach_edge_source_refs([edge], [edge_keys[1]]),
            ),
            timeout=30,
        )
        node_snapshot = (await adapter.get_node_delete_data([node_id]))[node_id]
        edge_snapshot = (await adapter.get_edge_delete_data([edge]))[edge]
        assert set(node_snapshot.source_ref_keys) == set(node_keys)
        assert set(edge_snapshot.source_ref_keys) == set(edge_keys)

        await asyncio.wait_for(
            asyncio.gather(
                adapter.remove_node_source_refs([node_id], [node_keys[0]]),
                second_adapter.remove_node_source_refs([node_id], [node_keys[1]]),
                adapter.remove_edge_source_refs([edge], [edge_keys[0]]),
                second_adapter.remove_edge_source_refs([edge], [edge_keys[1]]),
            ),
            timeout=30,
        )
        node_snapshot = (await adapter.get_node_delete_data([node_id]))[node_id]
        edge_snapshot = (await adapter.get_edge_delete_data([edge]))[edge]
        assert node_snapshot.source_ref_keys == []
        assert edge_snapshot.source_ref_keys == []
    finally:
        await second_adapter.close()


@pytest.mark.asyncio
async def test_concurrent_tag_removal_across_adapter_instances(adapter):
    """Two adapters stripping different tags from one node must both take effect.

    Row locks are the only thing keeping the read-modify-write over the JSONB
    property blob from losing one of the two updates.
    """
    second_adapter = PostgresDemoAdapter(adapter.db_uri)
    node_id = "concurrent-tag-node"

    try:
        await adapter.add_node(
            node_id, {"name": "Node", "type": "T", "belongs_to_set": ["keep", "drop_a", "drop_b"]}
        )
        # Open both connections up front: a cold connect costs more than a whole
        # transaction, which would serialize the two removals and test nothing.
        await second_adapter.get_node(node_id)

        await asyncio.wait_for(
            asyncio.gather(
                adapter.remove_belongs_to_set_tags(["drop_a"], node_ids=[node_id]),
                second_adapter.remove_belongs_to_set_tags(["drop_b"]),
            ),
            timeout=30,
        )

        node = await adapter.get_node(node_id)
        assert node["belongs_to_set"] == ["keep"]
    finally:
        await second_adapter.close()


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
async def test_get_neighbors_deduplicates_pairs_and_keeps_self_loops(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="neighbor-a", name="A", type="T"),
            _FakeDataPoint(id="neighbor-b", name="B", type="T"),
        ]
    )
    await adapter.add_edges(
        [
            ("neighbor-a", "neighbor-b", "R1", {}),
            ("neighbor-a", "neighbor-b", "R2", {}),
            ("neighbor-a", "neighbor-a", "LOOP", {}),
        ]
    )

    neighbors = await adapter.get_neighbors("neighbor-a")

    assert {node["id"] for node in neighbors} == {"neighbor-a", "neighbor-b"}


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


@pytest.mark.asyncio
async def test_get_neighborhood_with_asyncpg_seed_array(adapter):
    await adapter.add_nodes(
        [
            ("nh1", {"name": "A", "type": "Entity"}),
            ("nh2", {"name": "B", "type": "Entity"}),
        ]
    )
    await adapter.add_edges([("nh1", "nh2", "next", {})])

    nodes, edges = await adapter.get_neighborhood(["nh1"], depth=1)

    assert {node_id for node_id, _ in nodes} == {"nh1", "nh2"}
    assert ("nh1", "nh2", "next", {}) in edges


@pytest.mark.asyncio
async def test_neighborhood_depth_filter_and_boundary_rules(adapter):
    await adapter.add_nodes(
        [_FakeDataPoint(id=f"walk-{index}", name=str(index), type="T") for index in range(4)]
    )
    await adapter.add_edges(
        [
            ("walk-0", "walk-1", "walk", {}),
            ("walk-1", "walk-2", "walk", {}),
            ("walk-0", "walk-1", "other", {}),
            ("walk-1", "walk-3", "other", {}),
        ]
    )

    nodes, edges = await adapter.get_neighborhood(["walk-0"], depth=2, edge_types=["walk"])
    assert {node_id for node_id, _ in nodes} == {"walk-0", "walk-1", "walk-2"}
    assert {(source, target, name) for source, target, name, _ in edges} == {
        ("walk-0", "walk-1", "walk"),
        ("walk-0", "walk-1", "other"),
        ("walk-1", "walk-2", "walk"),
    }

    nodes, edges = await adapter.get_neighborhood(["walk-0", "walk-1", "missing"], depth=0)
    assert {node_id for node_id, _ in nodes} == {"walk-0", "walk-1"}
    assert {(source, target, name) for source, target, name, _ in edges} == {
        ("walk-0", "walk-1", "walk"),
        ("walk-0", "walk-1", "other"),
    }

    with pytest.raises(ValueError, match="non-negative"):
        await adapter.get_neighborhood(["walk-0"], depth=-1)


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
async def test_filtered_graph_projection_rules(adapter):
    await adapter.add_nodes(
        [
            ("projection-1", {"name": "1", "type": "Selected"}),
            ("projection-2", {"name": "2", "type": "Selected"}),
            ("projection-3", {"name": "3", "type": "Other"}),
            ("projection-isolated", {"name": "4", "type": "Selected"}),
        ]
    )
    await adapter.add_edges(
        [
            ("projection-1", "projection-2", "inside", {}),
            ("projection-2", "projection-3", "outside", {}),
        ]
    )

    nodes, edges = await adapter.get_id_filtered_graph_data(["projection-2"])
    assert {node_id for node_id, _ in nodes} == {
        "projection-1",
        "projection-2",
        "projection-3",
    }
    assert {relationship for _, _, relationship, _ in edges} == {"inside", "outside"}

    nodes, edges = await adapter.get_filtered_graph_data([{"type": ["Selected"]}])
    assert {node_id for node_id, _ in nodes} == {
        "projection-1",
        "projection-2",
        "projection-isolated",
    }
    assert [(source, target, name) for source, target, name, _ in edges] == [
        ("projection-1", "projection-2", "inside")
    ]

    with pytest.raises(ValueError, match="Invalid filter attribute"):
        await adapter.get_filtered_graph_data([{"properties": ["hidden"]}])


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

    assert node_ids == {"ns1", "ns2", "ns3"}
    assert {(source, target, name) for source, target, name, _ in edges} == {
        ("ns1", "ns2", "R"),
        ("ns1", "ns3", "R"),
    }


@pytest.mark.asyncio
async def test_nodeset_and_operator_uses_only_existing_primaries(adapter):
    await adapter.add_nodes(
        [
            ("primary-a", {"name": "Alpha", "type": "Entity"}),
            ("primary-b", {"name": "Beta", "type": "Entity"}),
            ("shared", {"name": "Shared", "type": "Other"}),
            ("one-side", {"name": "One", "type": "Other"}),
        ]
    )
    await adapter.add_edges(
        [
            ("primary-a", "shared", "R", {}),
            ("primary-b", "shared", "R", {}),
            ("primary-a", "one-side", "R", {}),
        ]
    )

    class Entity:
        pass

    nodes, _ = await adapter.get_nodeset_subgraph(
        Entity, ["Alpha", "Beta"], node_name_filter_operator="AND"
    )
    assert {node_id for node_id, _ in nodes} == {"primary-a", "primary-b", "shared"}

    nodes, _ = await adapter.get_nodeset_subgraph(
        Entity, ["Alpha", "Missing"], node_name_filter_operator="AND"
    )
    assert {node_id for node_id, _ in nodes} == {"primary-a", "shared", "one-side"}
    assert await adapter.get_nodeset_subgraph(Entity, ["Missing"], "AND") == ([], [])

    with pytest.raises(ValueError, match="must be 'OR' or 'AND'"):
        await adapter.get_nodeset_subgraph(Entity, ["Alpha"], "XOR")


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


@pytest.mark.asyncio
async def test_graph_metrics_components_and_optional_self_loops(adapter):
    await adapter.add_nodes(
        [_FakeDataPoint(id=f"metric-{index}", name=str(index), type="T") for index in range(6)]
    )
    await adapter.add_edges(
        [
            ("metric-0", "metric-1", "R", {}),
            ("metric-1", "metric-2", "R", {}),
            ("metric-3", "metric-4", "R", {}),
            ("metric-0", "metric-0", "LOOP", {}),
        ]
    )

    assert (await adapter.get_graph_metrics())["num_selfloops"] == -1
    metrics = await adapter.get_graph_metrics(include_optional=True)
    assert metrics["num_connected_components"] == 3
    assert metrics["sizes_of_connected_components"] == [3, 2, 1]
    assert metrics["num_selfloops"] == 1


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

    # Paginating must walk the same stable order, so each page continues the
    # previous one instead of repeating or skipping a triplet.
    pages = []
    for offset in range(2):
        page = await adapter.get_triplets_batch(offset=offset, limit=1)
        assert len(page) == 1
        pages.extend(page)
    assert pages == all_triplets


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
