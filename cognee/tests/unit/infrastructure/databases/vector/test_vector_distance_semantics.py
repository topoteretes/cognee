from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

try:
    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False

try:
    from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

    HAS_PGVECTOR = True
except ModuleNotFoundError:
    HAS_PGVECTOR = False

try:
    from cognee.infrastructure.databases.vector.chromadb.ChromaDBAdapter import ChromaDBAdapter

    HAS_CHROMADB = True
except ModuleNotFoundError:
    HAS_CHROMADB = False


class _DummyEmbeddingEngine:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class _AsyncContextManager:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeLanceQuery:
    def __init__(self, results):
        self.results = results
        self.distance_type_value = None
        self.where_value = None

    def distance_type(self, value):
        self.distance_type_value = value
        return self

    def where(self, value):
        self.where_value = value
        return self

    def select(self, _columns):
        return self

    def limit(self, _limit):
        return self

    async def to_list(self):
        return self.results


class _FakeLanceCollection:
    def __init__(self, results):
        self.results = results
        self.queries = []

    async def count_rows(self):
        return len(self.results)

    def vector_search(self, _query_vector):
        query = _FakeLanceQuery(self.results)
        self.queries.append(query)
        return query


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, _query):
        return SimpleNamespace(all=lambda: self._rows)


class _FakeQuery:
    def where(self, *_args, **_kwargs):
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_lancedb_search_returns_raw_distance_and_uses_cosine():
    adapter = LanceDBAdapter(
        url="memory://", api_key=None, embedding_engine=_DummyEmbeddingEngine()
    )

    item_id = str(uuid4())
    collection = _FakeLanceCollection(
        [
            {"id": item_id, "payload": {"text": "x"}, "_distance": 1.42},
            {"id": str(uuid4()), "payload": {"text": "y"}, "_distance": 0.21},
        ]
    )
    adapter.get_collection = AsyncMock(return_value=collection)

    results = await adapter.search(
        collection_name="Entity_name",
        query_vector=[0.1, 0.2, 0.3],
        include_payload=True,
    )

    assert len(results) == 2
    assert [result.score for result in results] == [1.42, 0.21]
    assert any(result.score > 1.0 for result in results)
    assert collection.queries[0].distance_type_value == "cosine"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb extra is not installed")
async def test_lancedb_search_with_nodeset_filter_uses_cosine():
    adapter = LanceDBAdapter(
        url="memory://", api_key=None, embedding_engine=_DummyEmbeddingEngine()
    )
    item_id = str(uuid4())
    collection = _FakeLanceCollection(
        [{"id": item_id, "payload": {"belongs_to_set": ["A"]}, "_distance": 1.11}]
    )
    adapter.get_collection = AsyncMock(return_value=collection)

    results = await adapter.search(
        collection_name="Entity_name",
        query_vector=[0.1, 0.2, 0.3],
        node_name=["A"],
        include_payload=True,
    )

    assert [result.score for result in results] == [1.11]
    assert collection.queries[0].distance_type_value == "cosine"
    assert "array_has_any" in collection.queries[0].where_value


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb extra is not installed")
async def test_chromadb_search_and_batch_return_raw_distance():
    adapter = ChromaDBAdapter(
        url="http://unused", api_key=None, embedding_engine=_DummyEmbeddingEngine()
    )

    ids = [str(uuid4()), str(uuid4())]
    collection = SimpleNamespace()
    collection.query = AsyncMock(
        side_effect=[
            {
                "ids": [[ids[0]]],
                "metadatas": [[{"text": "a"}]],
                "distances": [[1.31]],
            },
            {
                "ids": [[ids[0], ids[1]], [ids[1]]],
                "metadatas": [[{"text": "a"}, {"text": "b"}], [{"text": "c"}]],
                "distances": [[1.7, 0.2], [1.4]],
            },
        ]
    )
    adapter.get_collection = AsyncMock(return_value=collection)
    adapter.embed_data = AsyncMock(return_value=[[0.1], [0.2]])

    single = await adapter.search(
        collection_name="Entity_name",
        query_vector=[0.1, 0.2, 0.3],
    )
    batch = await adapter.batch_search(
        collection_name="Entity_name",
        query_texts=["a", "b"],
        limit=2,
    )

    assert [r.score for r in single] == [1.31]
    assert [r.score for r in batch[0]] == [1.7, 0.2]
    assert any(r.score > 1.0 for r in single + batch[0] + batch[1])


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_PGVECTOR, reason="pgvector extra is not installed")
async def test_pgvector_search_returns_raw_distance(monkeypatch):
    import cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter as pg_module

    monkeypatch.setattr(pg_module, "select", lambda *args, **kwargs: _FakeQuery())

    adapter = PGVectorAdapter.__new__(PGVectorAdapter)
    adapter.embedding_engine = _DummyEmbeddingEngine()
    adapter.get_table = AsyncMock(
        return_value=SimpleNamespace(
            c=SimpleNamespace(
                id="id_col",
                vector=SimpleNamespace(
                    cosine_distance=lambda _v: SimpleNamespace(label=lambda _l: "x")
                ),
            )
        )
    )
    adapter.get_async_session = lambda: _AsyncContextManager(  # noqa: E731
        _FakeSession(
            [
                SimpleNamespace(id=str(uuid4()), similarity=1.19),
                SimpleNamespace(id=str(uuid4()), similarity=0.51),
            ]
        )
    )

    results = await PGVectorAdapter.search(
        adapter,
        collection_name="Entity_name",
        query_vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert [r.score for r in results] == [1.19, 0.51]
    assert any(r.score > 1.0 for r in results)


@pytest.mark.asyncio
async def test_cross_collection_scores_are_globally_comparable() -> None:
    """Regression test for GH-2030: per-collection normalization must NOT be applied.

    When brute_force_triplet_search queries multiple collections the raw cosine
    distances returned by each adapter are mapped directly onto graph nodes.  If
    scores were min-max-normalised *per collection* a node with distance 0.1 from
    Entity_name and a node with distance 0.5 from TextSummary_text would both end
    up with score 0.0 (the local minimum), making them appear equally relevant.
    Instead the raw values must be preserved so that cross-collection ranking is
    correct.

    This test verifies the invariant via CogneeGraph.map_vector_distances_to_graph_nodes()
    without touching any real database.
    """
    from typing import Any, Dict

    from cognee.modules.graph.cognee_graph.CogneeGraph import CogneeGraph
    from cognee.modules.graph.cognee_graph.CogneeGraphElements import Node, Edge

    # Two nodes, one matched in each collection with different raw distances.
    node_a_id = str(uuid4())
    node_b_id = str(uuid4())

    class _ScoredResult:
        def __init__(self, id_: str, score: float) -> None:
            self.id: str = id_
            self.score: float = score
            self.payload: Dict[str, Any] = {}

    # node_a appears in Entity_name with a very good (low) distance
    # node_b appears in TextSummary_text with a worse distance
    node_distances = {
        "Entity_name": [_ScoredResult(node_a_id, 0.1)],
        "TextSummary_text": [_ScoredResult(node_b_id, 0.5)],
    }

    # Build a minimal graph with both nodes connected by a single edge
    graph = CogneeGraph()
    node_a = Node(node_id=node_a_id, attributes={"name": "A", "type": "Entity"})
    node_b = Node(node_id=node_b_id, attributes={"name": "B", "type": "Entity"})
    edge = Edge(
        node1=node_a,
        node2=node_b,
        attributes={"relationship_name": "related_to"},
    )
    graph.nodes[node_a_id] = node_a
    graph.nodes[node_b_id] = node_b
    graph.edges.append(edge)
    graph.triplet_distance_penalty = 6.5

    await graph.map_vector_distances_to_graph_nodes(node_distances=node_distances)

    distance_a = node_a.attributes["vector_distance"][0]
    distance_b = node_b.attributes["vector_distance"][0]

    # Raw distances must be preserved — no per-collection min-max normalization.
    assert distance_a == pytest.approx(0.1), (
        f"Expected raw distance 0.1 for node_a, got {distance_a}. "
        "Per-collection normalization would collapse this to 0.0."
    )
    assert distance_b == pytest.approx(0.5), (
        f"Expected raw distance 0.5 for node_b, got {distance_b}. "
        "Per-collection normalization would collapse this to 0.0."
    )
    # The closer node must rank better (lower score = more relevant).
    assert distance_a < distance_b, (
        "Cross-collection ranking is broken: node_a (distance 0.1) should rank "
        "better than node_b (distance 0.5) regardless of which collection each came from."
    )
