"""Graph-first scoring must preserve exact ranking on each local vector backend.

Ladybug holds the real scoped graph. LanceDB (in-process and subprocess), Turso
and optionally PGVector hold its vectors, including closer out-of-scope distractors.
Set TEST_PGVECTOR_URL to exercise Postgres; no LLM or network embedding is needed.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import BaseModel
from sqlalchemy import text

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import LanceDBAdapter
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    _get_top_triplet_importances,
    brute_force_triplet_search,
    get_memory_fragment,
)
from cognee.modules.retrieval.utils.node_edge_vector_search import NodeEdgeVectorSearch


class ScopeAnchor(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class ScopeValue(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


class _Payload(BaseModel):
    text: str


class _Embeddings:
    def __init__(self):
        self.calls = []

    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0, 0.0] if value == "first" else [0.0, 1.0, 0.0] for value in texts]


@pytest_asyncio.fixture(params=["lancedb", "lancedb-subprocess", "turso", "pgvector"])
async def stores(request, tmp_path):
    embeddings = _Embeddings()
    schema = None
    if request.param == "pgvector":
        url = os.getenv("TEST_PGVECTOR_URL")
        if not url:
            pytest.skip("TEST_PGVECTOR_URL not set")
        pytest.importorskip("pgvector")
        from cognee.infrastructure.databases.vector.pgvector.PGVectorAdapter import PGVectorAdapter

        schema = f"scope_test_{uuid4().hex}"
        vector = PGVectorAdapter(url, None, embeddings, schema=schema)
        async with vector.get_async_session() as session:
            await session.execute(text(f'CREATE SCHEMA "{schema}"'))
            await session.commit()
    elif request.param == "turso":
        pytest.importorskip("turso")
        from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import (
            TursoVectorAdapter,
        )

        vector = TursoVectorAdapter(str(tmp_path / "vectors.db"), None, embeddings)
    elif request.param == "lancedb-subprocess":
        vector = LanceDBAdapter.create_subprocess(str(tmp_path / "vectors"), None, embeddings)
    else:
        vector = LanceDBAdapter(str(tmp_path / "vectors"), None, embeddings)
    graph = LadybugAdapter(
        str(tmp_path / "graph"), kuzu_buffer_pool_size=64 * 1024 * 1024, kuzu_num_threads=2
    )
    try:
        yield SimpleNamespace(vector=vector, graph=graph), embeddings
    finally:
        await graph.close()
        if schema:
            async with vector.get_async_session() as session:
                await session.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
                await session.commit()
        await vector.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_graph_first_scoring_matches_exhaustive_scoring(stores, batch):
    engine, embeddings = stores
    anchor = ScopeAnchor(name="customer:last_name")
    values = [ScopeValue(name=value) for value in ["first", "second", "unembedded", "outside"]]
    await engine.graph.add_nodes([anchor, *values])
    await engine.graph.add_edges(
        [
            (str(anchor.id), str(value.id), "has_value", {"edge_text": f"contains {value.name}"})
            for value in values[:3]
        ]
    )
    collections = ["ScopeValue_name", "EdgeType_relationship_name"]
    vectors = [
        {"id": str(value.id), "vector": vector, "payload": {"text": value.name}}
        for value, vector in zip(
            [values[0], values[1], values[3]],
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        )
    ]
    edge_vectors = [
        {
            "id": str(EdgeType.id_for(f"contains {value.name}")),
            "vector": [0.0, 1.0, 0.0],
            "payload": {"text": value.name},
        }
        for value in values[:2]
    ]
    await engine.vector.upsert_raw_vectors(collections[0], vectors, payload_schema=_Payload)
    await engine.vector.upsert_raw_vectors(collections[1], edge_vectors, payload_schema=_Payload)

    query_args = {"query_batch": ["first", "second"]} if batch else {"query": "first"}
    full_search = NodeEdgeVectorSearch(vector_engine=engine.vector)
    await full_search.embed_and_retrieve_distances(
        **query_args, collections=collections, wide_search_limit=None
    )
    fragment = await get_memory_fragment(
        node_type=ScopeAnchor, node_name=[anchor.name], graph_engine=engine.graph
    )
    expected = await _get_top_triplet_importances(
        fragment,
        full_search,
        None,
        ScopeAnchor,
        [anchor.name],
        "OR",
        6.5,
        0.0,
        None,
        6,
        query_list_length=2 if batch else None,
    )
    with (
        patch.object(engine.vector, "search", side_effect=AssertionError("unscoped search")),
        patch.object(engine.vector, "batch_search", side_effect=AssertionError("unscoped batch")),
        patch.object(engine.graph, "get_graph_data", side_effect=AssertionError("full graph read")),
    ):
        actual = await brute_force_triplet_search(
            **query_args,
            node_type=ScopeAnchor,
            node_name=[anchor.name],
            collections=collections,
            top_k=6,
            feedback_influence=0.0,
            unified_engine=engine,
        )
    pairs = zip(actual, expected) if batch else [(actual, expected)]
    for optimized, exhaustive in pairs:

        def logical_order(edges):
            # Mirrored Ladybug edges have equal scores; their relative order is unspecified.
            return list(
                dict.fromkeys(tuple(sorted([edge.node1.id, edge.node2.id])) for edge in edges)
            )

        assert logical_order(optimized) == logical_order(exhaustive)
        expected_by_key = {(edge.node1.id, edge.node2.id): edge for edge in exhaustive}
        assert {(edge.node1.id, edge.node2.id) for edge in optimized} == set(expected_by_key)
        assert len(optimized) == 6  # Ladybug returns both orientations of each edge.
        assert all(str(values[3].id) not in [edge.node1.id, edge.node2.id] for edge in optimized)
        for optimized_edge in optimized:
            exhaustive_edge = expected_by_key[(optimized_edge.node1.id, optimized_edge.node2.id)]
            for optimized_node, exhaustive_node in [
                (optimized_edge.node1, exhaustive_edge.node1),
                (optimized_edge.node2, exhaustive_edge.node2),
            ]:
                assert optimized_node.attributes["vector_distance"] == pytest.approx(
                    exhaustive_node.attributes["vector_distance"], abs=1e-6
                )
            assert optimized_edge.attributes["vector_distance"] == pytest.approx(
                exhaustive_edge.attributes["vector_distance"], abs=1e-6
            )

    embeddings.calls.clear()
    assert (
        await brute_force_triplet_search(
            query="first",
            node_type=ScopeAnchor,
            node_name=["missing"],
            unified_engine=engine,
        )
        == []
    )
    assert embeddings.calls == []
