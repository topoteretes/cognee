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

    # Build a fresh SQLAlchemyAdapter per test to avoid stale event loop
    # issues from lru_cached global engines across pytest-asyncio boundaries
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    relational_engine = SQLAlchemyAdapter(connection_string)
    graph_adapter = PostgresAdapter(relational_engine=relational_engine)

    # Give PGVectorAdapter the same engine/sessionmaker directly to
    # avoid it calling get_relational_engine() which returns a cached
    # instance tied to a previous event loop
    vector_adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    vector_adapter.api_key = None
    vector_adapter.embedding_engine = get_embedding_engine()
    vector_adapter.db_uri = connection_string
    vector_adapter.VECTOR_DB_LOCK = __import__("asyncio").Lock()
    vector_adapter.engine = relational_engine.engine
    vector_adapter.sessionmaker = relational_engine.sessionmaker
    from pgvector.sqlalchemy import Vector

    vector_adapter.Vector = Vector

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

    # Drop any vector collection tables created during the test to
    # avoid polluting the shared connection pool for the next fixture
    try:
        async with relational_engine.get_async_session() as session:
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
        assert "name" in result.payload
        if str(result.id) == id1:
            assert result.payload["name"] == "Quantum Computing"
        elif str(result.id) == id3:
            assert result.payload["name"] == "Neural Networks"

    # -- Verify vector search returns relevant results --
    search_results = await adapter.search(
        "TestEntity_name",
        query_text="quantum physics computing",
        limit=3,
        include_payload=True,
    )
    assert len(search_results) > 0
    # The closest result should be "Quantum Computing"
    assert search_results[0].payload["name"] == "Quantum Computing"


# -- Tests: combined writes with vector collections (run last) --


@pytest.mark.asyncio
async def test_add_edges_with_vectors(adapter):
    """Edges should be inserted into graph and edge types into vector table.

    Note: this test passes in isolation but may fail when run in suite
    due to PGVectorAdapter connection pool state from prior tests.
    This is a pre-existing PGVectorAdapter issue, not a hybrid adapter bug.
    """
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
