"""Real local graph/vector search; only external LLM and embeddings are deterministic fakes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
    IndexSchema,
    LanceDBAdapter,
)
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.retrieval.hybrid_decomposition_retriever import HybridDecompositionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition

QUESTION = "What did Acme order, and when is delivery due?"
SUBQUERY = "When is Acme's delivery due?"
ORDER = "Acme ordered 40 units."
DELIVERY = "Acme's delivery is due on 28 April."

# Edges of the entity/fact case, with the vector each is stored under. The question sits
# on the first axis and the subquery on the second, so each ranks a different edge first.
ORDERED_EDGE = "Acme ordered 40 units from Warehouse"
DELIVERY_EDGE = "Acme delivery is handled by Carrier"
PALLETS_EDGE = "Carrier collects pallets from Warehouse nightly"
GRAPH_VECTORS = {
    QUESTION: [1.0, 0.0, 0.0],
    SUBQUERY: [0.0, 1.0, 0.0],
    ORDERED_EDGE: [1.0, 0.0, 0.0],
    DELIVERY_EDGE: [0.6, 0.8, 0.0],
    PALLETS_EDGE: [0.5, 0.5, 0.7],
    "Acme": [0.7, 0.7, 0.0],
}


class FixedEmbeddings:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [
            [0.0, 1.0, 0.0] if text in (SUBQUERY, DELIVERY) else [1.0, 0.0, 0.0] for text in texts
        ]


class GraphEmbeddings:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [GRAPH_VECTORS[text] for text in texts]


@pytest.mark.asyncio
async def test_local_stores_union_entity_edges_and_drop_facts_shown_as_edges(tmp_path):
    """Entity and fact lanes on, against real stores. With one edge and one fact per
    retrieval, the question shows the order edge and returns the delivery edge as a fact;
    the subquery shows the delivery edge. The union keeps both edges on the entity and
    drops the fact that is now one of them."""
    graph = LadybugAdapter(db_path=str(tmp_path / "graph"))
    vector = LanceDBAdapter(
        url=str(tmp_path / "vectors"), api_key=None, embedding_engine=GraphEmbeddings()
    )
    unified = SimpleNamespace(graph=graph, vector=vector)
    acme, warehouse, carrier = (
        DataPoint(name=name, type="Entity") for name in ("Acme", "Warehouse", "Carrier")
    )
    edges = [
        (acme, warehouse, "ordered", ORDERED_EDGE),
        (acme, carrier, "delivery", DELIVERY_EDGE),
        (carrier, warehouse, "collects", PALLETS_EDGE),
    ]
    try:
        for node in (acme, warehouse, carrier):
            await graph.add_node(node)
        for source, target, relationship, text in edges:
            await graph.add_edge(str(source.id), str(target.id), relationship, {"edge_text": text})
        await vector.create_collection("Entity_name", payload_schema=IndexSchema)
        await vector.create_data_points("Entity_name", [IndexSchema(id=str(acme.id), text="Acme")])
        await vector.create_collection("EdgeType_relationship_name", payload_schema=IndexSchema)
        await vector.create_data_points(
            "EdgeType_relationship_name",
            [IndexSchema(id=str(EdgeType.id_for(text)), text=text) for *_, text in edges],
        )
        options = {
            "chunks_top_k": 0,
            "entities_top_k": 1,
            "facts_top_k": 1,
            "max_edges_per_entity": 1,
            "text_summaries_top_k": 0,
        }
        with (
            patch(
                "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
                AsyncMock(return_value=unified),
            ),
            patch(
                "cognee.modules.retrieval.utils.query_decomposition.LLMGateway.acreate_structured_output",
                AsyncMock(return_value=QueryDecomposition(subqueries=[SUBQUERY])),
            ),
        ):
            baseline = await HybridRetriever(**options).get_retrieved_objects(query=QUESTION)
            decomposed = HybridDecompositionRetriever(**options)
            expanded = await decomposed.get_retrieved_objects(query=QUESTION)
            context = await decomposed.get_context_from_objects(
                query=QUESTION, retrieved_objects=expanded
            )

        def edge_texts(result):
            return [[edge["text"] for edge in entity["edges"]] for entity in result["entities"]]

        def fact_texts(result):
            return [fact["text"] for fact in result["facts"]]

        assert edge_texts(baseline) == [[ORDERED_EDGE]]
        assert fact_texts(baseline) == [DELIVERY_EDGE]
        # Both retrievals' edges, more than max_edges_per_entity allows one retrieval.
        assert edge_texts(expanded) == [[ORDERED_EDGE, DELIVERY_EDGE]]
        # The delivery fact is now an entity edge, so only the subquery's other fact stays.
        assert fact_texts(expanded) == [PALLETS_EDGE]
        assert context.count(DELIVERY_EDGE) == 1
    finally:
        await vector.close()
        await graph.close()


@pytest.mark.asyncio
async def test_local_stores_expand_coverage_and_preserve_chunk_metadata(tmp_path):
    graph = LadybugAdapter(db_path=str(tmp_path / "graph"))
    vector = LanceDBAdapter(
        url=str(tmp_path / "vectors"), api_key=None, embedding_engine=FixedEmbeddings()
    )
    unified = SimpleNamespace(graph=graph, vector=vector)
    chunks = [
        IndexSchema(id=str(uuid4()), text=text, external_metadata='{"source_id":"order-A"}')
        for text in (ORDER, DELIVERY)
    ]
    try:
        await graph.add_node(DataPoint(name="Acme", type="Entity"))
        await vector.create_collection("DocumentChunk_text", payload_schema=IndexSchema)
        await vector.create_data_points("DocumentChunk_text", chunks)
        options = {
            "chunks_top_k": 1,
            "entities_top_k": 0,
            "facts_top_k": 0,
            "text_summaries_top_k": 0,
            "include_external_metadata": True,
            "external_metadata_keys": ["source_id"],
        }
        plain = HybridRetriever(**options)
        decomposed = HybridDecompositionRetriever(**options)
        with (
            patch(
                "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
                AsyncMock(return_value=unified),
            ),
            patch(
                "cognee.modules.retrieval.utils.query_decomposition.LLMGateway.acreate_structured_output",
                AsyncMock(return_value=QueryDecomposition(subqueries=[SUBQUERY])),
            ) as planner,
        ):
            baseline = await plain.get_retrieved_objects(query=QUESTION)
            expanded = await decomposed.get_retrieved_objects(query=QUESTION)
            context = await decomposed.get_context_from_objects(
                query=QUESTION, retrieved_objects=expanded
            )
        assert [str(hit.id) for hit in baseline["chunks"]] == [chunks[0].id]
        assert {str(hit.id) for hit in expanded["chunks"]} == {chunk.id for chunk in chunks}
        assert ORDER in context and DELIVERY in context and "order-A" in context
        assert all(
            hit.payload["external_metadata"]["source_id"] == "order-A" for hit in expanded["chunks"]
        )
        assert QUESTION in planner.await_args.kwargs["text_input"]
    finally:
        await vector.close()
        await graph.close()
