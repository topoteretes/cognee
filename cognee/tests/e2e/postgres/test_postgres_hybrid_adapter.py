"""Unit tests for the Postgres hybrid adapter.

Requires a running Postgres instance with pgvector extension.
Connection defaults: DB_HOST=localhost, DB_PORT=5432,
DB_USERNAME=cognee, DB_PASSWORD=cognee, DB_NAME=cognee_db.
"""

import os
import pytest
import pytest_asyncio

from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter
from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.infrastructure.databases.hybrid.postgres.adapter import PostgresHybridAdapter


# -- Session-scoped event loop so async engines stay on a single loop.


@pytest.fixture(scope="session")
def event_loop():
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# -- Fixture --


@pytest_asyncio.fixture
async def adapter():
    """Create hybrid adapter backed by real Postgres with pgvector."""
    db_username = os.environ.get("DB_USERNAME", "cognee")
    db_password = os.environ.get("DB_PASSWORD", "cognee")
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME", "cognee_db")

    connection_string = (
        f"postgresql+asyncpg://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    graph_adapter = PostgresAdapter(connection_string=connection_string)
    vector_adapter = PGVectorAdapter(
        connection_string=connection_string,
        api_key=None,
        embedding_engine=get_embedding_engine(),
    )

    # Enable pgvector extension -- normally done by cognee's create_db_and_tables()
    # during startup, but this test constructs adapters directly.
    from sqlalchemy import text as sa_text

    async with graph_adapter.engine.begin() as conn:
        await conn.execute(sa_text("CREATE EXTENSION IF NOT EXISTS vector"))

    a = PostgresHybridAdapter(
        graph_adapter=graph_adapter,
        vector_adapter=vector_adapter,
    )
    await a.initialize()
    await a.delete_graph()

    yield a

    # Clean up graph tables; ignore errors from tables already dropped
    try:
        await a.delete_graph()
    except Exception:
        pass

    # Drop any vector collection tables created during the test
    try:
        async with graph_adapter.sessionmaker() as session:
            from sqlalchemy import text as sa_text

            for table_name in [
                "EdgeType_relationship_name",
                "TestEntity_name",
                "test_hybrid_collection",
            ]:
                await session.execute(sa_text(f'DROP TABLE IF EXISTS "{table_name}" CASCADE'))
            await session.commit()
    except Exception:
        pass


# -- Helpers --


class _FakeDataPoint:
    """Minimal DataPoint-like object for testing (no vector indexing)."""

    def __init__(self, id, name="", type="", **extra):
        self._data = {"id": str(id), "name": name, "type": type, **extra}
        self.id = str(id)
        self.metadata = {"index_fields": []}
        self.belongs_to_set = []

    def model_dump(self):
        return dict(self._data)

    def model_copy(self):
        return _FakeDataPoint(**self._data)


class TestEntity:
    """DataPoint-like object with an indexable 'name' field for vector embedding.

    Class name becomes the collection prefix: TestEntity_name.
    """

    def __init__(self, id, name="", type="", belongs_to_set=None, **extra):
        self._data = {"id": str(id), "name": name, "type": type, **extra}
        self.id = str(id)
        self.name = name
        self.type = type
        self.metadata = {"index_fields": ["name"]}
        self.belongs_to_set = belongs_to_set or []

    def model_dump(self):
        return dict(self._data)

    def model_copy(self):
        cp = TestEntity(**self._data)
        cp.metadata = dict(self.metadata)
        return cp


# -- Tests: graph delegation --


@pytest.mark.asyncio
async def test_is_empty_on_fresh_db(adapter):
    assert await adapter.is_empty() is True


@pytest.mark.asyncio
async def test_add_and_get_node(adapter):
    node = _FakeDataPoint(id="h1", name="Alice", type="Person")
    await adapter.add_node(node)
    result = await adapter.get_node("h1")
    assert result is not None
    assert result["name"] == "Alice"


@pytest.mark.asyncio
async def test_add_nodes_and_edges(adapter):
    nodes = [
        _FakeDataPoint(id="ha", name="A", type="T"),
        _FakeDataPoint(id="hb", name="B", type="T"),
    ]
    await adapter.add_nodes(nodes)
    await adapter.add_edge("ha", "hb", "KNOWS")

    assert await adapter.has_edge("ha", "hb", "KNOWS") is True
    neighbors = await adapter.get_neighbors("ha")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "hb"


@pytest.mark.asyncio
async def test_get_graph_data(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="hg1", name="X", type="T"),
            _FakeDataPoint(id="hg2", name="Y", type="T"),
        ]
    )
    await adapter.add_edge("hg1", "hg2", "R")

    nodes, edges = await adapter.get_graph_data()
    assert len(nodes) == 2
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_delete_graph(adapter):
    await adapter.add_nodes([_FakeDataPoint(id="hd1", name="Z", type="T")])
    await adapter.delete_graph()
    assert await adapter.is_empty() is True


# -- Tests: vector delegation --


@pytest.mark.asyncio
async def test_has_collection_false(adapter):
    assert await adapter.has_collection("nonexistent_collection_xyz") is False


@pytest.mark.asyncio
async def test_embed_data(adapter):
    """Vector delegation: embedding engine is accessible."""
    vectors = await adapter.embed_data(["hello world"])
    assert len(vectors) == 1
    assert len(vectors[0]) > 0


# -- Tests: query raises --


@pytest.mark.asyncio
async def test_query_raises_not_implemented(adapter):
    with pytest.raises(NotImplementedError):
        await adapter.query("MATCH (n) RETURN n")


# -- Tests: combined write methods --
# Note: tests that create vector collections (add_edges_with_vectors,
# add_nodes_with_vectors with index fields) are grouped at the end
# because PGVectorAdapter's connection pool has ordering sensitivity
# when creating/dropping tables across tests.


@pytest.mark.asyncio
async def test_add_nodes_with_vectors_graph_only(adapter):
    """Data points with no index_fields should still insert into graph."""
    nodes = [
        _FakeDataPoint(id="nv1", name="NoVec", type="T"),
        _FakeDataPoint(id="nv2", name="NoVec2", type="T"),
    ]
    await adapter.add_nodes_with_vectors(nodes)

    result = await adapter.get_nodes(["nv1", "nv2"])
    assert len(result) == 2


# -- Tests: combined delete methods --


@pytest.mark.asyncio
async def test_delete_nodes_with_vectors(adapter):
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="dn1", name="A", type="T"),
            _FakeDataPoint(id="dn2", name="B", type="T"),
        ]
    )
    await adapter.add_edge("dn1", "dn2", "R")

    await adapter.delete_nodes_with_vectors(["dn1", "dn2"])

    assert await adapter.get_node("dn1") is None
    assert await adapter.get_node("dn2") is None
    assert await adapter.has_edge("dn1", "dn2", "R") is False


# -- Tests: content verification across graph and vector --


@pytest.mark.asyncio
async def test_combined_write_content_integrity(adapter):
    """Verify that data written via combined methods is stored correctly
    in both graph and vector tables, with matching IDs and intact contents.
    """
    from uuid import uuid5, NAMESPACE_OID

    # Generate stable UUIDs from names (pgvector tables require UUID IDs)
    id1 = str(uuid5(NAMESPACE_OID, "Quantum Computing"))
    id2 = str(uuid5(NAMESPACE_OID, "Machine Learning"))
    id3 = str(uuid5(NAMESPACE_OID, "Neural Networks"))

    # Add nodes with indexable fields (will create TestEntity_name collection)
    nodes = [
        TestEntity(id=id1, name="Quantum Computing", type="Topic"),
        TestEntity(id=id2, name="Machine Learning", type="Topic"),
        TestEntity(id=id3, name="Neural Networks", type="Topic"),
    ]
    await adapter.add_nodes_with_vectors(nodes)

    # Add edges with properties
    edges = [
        (id1, id2, "RELATED_TO", {"weight": 0.8, "edge_text": "quantum ML"}),
        (id2, id3, "CONTAINS", {"weight": 0.9, "edge_text": "neural nets in ML"}),
    ]
    await adapter.add_edges_with_vectors(edges)

    # -- Verify graph node contents --
    node1 = await adapter.get_node(id1)
    assert node1["id"] == id1
    assert node1["name"] == "Quantum Computing"
    assert node1["type"] == "Topic"

    node3 = await adapter.get_node(id3)
    assert node3["name"] == "Neural Networks"

    # -- Verify graph edge contents --
    connections = await adapter.get_connections(id1)
    rel_names = {conn[1]["relationship_name"] for conn in connections}
    assert "RELATED_TO" in rel_names

    # Verify edge properties preserved
    for _, edge, _ in connections:
        if edge["relationship_name"] == "RELATED_TO":
            assert edge.get("weight") == 0.8
            assert edge.get("edge_text") == "quantum ML"

    # -- Verify graph data completeness --
    all_nodes, all_edges = await adapter.get_graph_data()
    node_ids = {n[0] for n in all_nodes}
    assert node_ids == {id1, id2, id3}
    assert len(all_edges) == 2

    # -- Verify vector collection was created --
    assert await adapter.has_collection("TestEntity_name") is True
    assert await adapter.has_collection("EdgeType_relationship_name") is True

    # -- Verify vector entries have correct IDs --
    vector_results = await adapter.retrieve("TestEntity_name", [id1, id2, id3])
    vector_ids = {str(r.id) for r in vector_results}
    assert id1 in vector_ids
    assert id2 in vector_ids
    assert id3 in vector_ids

    # -- Verify vector payloads contain original data --
    for result in vector_results:
        assert result.payload is not None
        assert "text" in result.payload
        if str(result.id) == id1:
            assert result.payload["text"] == "Quantum Computing"
        elif str(result.id) == id3:
            assert result.payload["text"] == "Neural Networks"

    # -- Verify vector search returns relevant results --
    search_results = await adapter.search(
        "TestEntity_name",
        query_text="quantum physics computing",
        limit=3,
        include_payload=True,
    )
    assert len(search_results) > 0
    # The closest result should be "Quantum Computing"
    assert search_results[0].payload["text"] == "Quantum Computing"


@pytest.mark.asyncio
async def test_hybrid_payload_matches_separate_path(adapter):
    """Vector payloads from add_nodes_with_vectors + add_edges_with_vectors
    must be identical to what the separate add_nodes + index_data_points
    and add_edges + index_data_points paths produce."""
    from uuid import uuid5, NAMESPACE_OID

    # Build two parallel graphs with the same structure but different IDs
    def make_ids(prefix):
        return {k: str(uuid5(NAMESPACE_OID, f"{prefix}_{k}")) for k in ("physics", "math", "cs")}

    h = make_ids("hybrid")
    s = make_ids("separate")

    def make_nodes(ids):
        return [
            TestEntity(id=ids["physics"], name="Quantum Physics", type="Science"),
            TestEntity(id=ids["math"], name="Linear Algebra", type="Mathematics"),
            TestEntity(id=ids["cs"], name="Machine Learning", type="Computer Science"),
        ]

    def make_edges(ids):
        return [
            (ids["physics"], ids["math"], "REQUIRES", {"weight": 0.9}),
            (ids["math"], ids["cs"], "ENABLES", {"weight": 0.7}),
            (ids["physics"], ids["cs"], "INSPIRES", {"weight": 0.5}),
        ]

    # -- Path A: hybrid --
    await adapter.add_nodes_with_vectors(make_nodes(h))
    await adapter.add_edges_with_vectors(make_edges(h))

    # -- Path B: separate --
    sep_nodes = make_nodes(s)
    await adapter._graph.add_nodes(sep_nodes)
    await adapter._vector.index_data_points("TestEntity", "name", sep_nodes)
    sep_edges = make_edges(s)
    await adapter._graph.add_edges(sep_edges)
    from cognee.tasks.storage.index_graph_edges import index_graph_edges

    await index_graph_edges(sep_edges, vector_engine=adapter._vector)

    # -- Compare node payloads --
    hybrid_node_results = await adapter.retrieve("TestEntity_name", list(h.values()))
    separate_node_results = await adapter.retrieve("TestEntity_name", list(s.values()))
    assert len(hybrid_node_results) == 3
    assert len(separate_node_results) == 3

    # Sort by text so we compare matching nodes
    hybrid_node_results.sort(key=lambda r: r.payload["text"])
    separate_node_results.sort(key=lambda r: r.payload["text"])

    skip_keys = {"id", "created_at", "updated_at"}
    for h_res, s_res in zip(hybrid_node_results, separate_node_results):
        assert set(h_res.payload.keys()) == set(s_res.payload.keys()), (
            f"node key mismatch: hybrid={set(h_res.payload.keys())} "
            f"separate={set(s_res.payload.keys())}"
        )
        for key in h_res.payload:
            if key in skip_keys:
                continue
            assert h_res.payload[key] == s_res.payload[key], (
                f"node key {key!r}: hybrid={h_res.payload[key]!r} "
                f"!= separate={s_res.payload[key]!r}"
            )

    # -- Compare edge type payloads --
    hybrid_edge_results = await adapter.search(
        "EdgeType_relationship_name",
        query_text="REQUIRES ENABLES INSPIRES",
        limit=10,
        include_payload=True,
    )
    # Both paths write to the same EdgeType collection; verify at least 3 types
    edge_texts = {r.payload["text"] for r in hybrid_edge_results}
    assert {"REQUIRES", "ENABLES", "INSPIRES"} <= edge_texts


# -- Tests: combined writes with vector collections (run last) --


@pytest.mark.asyncio
async def test_add_edges_with_vectors(adapter):
    """Edges should be inserted into graph and edge types into vector table."""
    await adapter.add_nodes(
        [
            _FakeDataPoint(id="ev1", name="A", type="T"),
            _FakeDataPoint(id="ev2", name="B", type="T"),
            _FakeDataPoint(id="ev3", name="C", type="T"),
        ]
    )
    edges = [
        ("ev1", "ev2", "KNOWS", {}),
        ("ev2", "ev3", "LIKES", {}),
        ("ev1", "ev3", "KNOWS", {}),
    ]
    await adapter.add_edges_with_vectors(edges)

    # Graph edges present
    assert await adapter.has_edge("ev1", "ev2", "KNOWS") is True
    assert await adapter.has_edge("ev2", "ev3", "LIKES") is True

    # EdgeType vector collection should exist
    assert await adapter.has_collection("EdgeType_relationship_name") is True


@pytest.mark.asyncio
async def test_delete_graph_clears_nodes_and_edges(adapter):
    """Graph-side prune: truncate graph tables."""
    await adapter.add_nodes([_FakeDataPoint(id="pa1", name="A", type="T")])
    await adapter.add_nodes([_FakeDataPoint(id="pa2", name="B", type="T")])
    await adapter.add_edge("pa1", "pa2", "R")
    await adapter.delete_graph()
    assert await adapter.is_empty() is True
    assert await adapter.has_edge("pa1", "pa2", "R") is False
