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
from cognee.modules.retrieval.hybrid_decomposition_retriever import HybridDecompositionRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition

QUESTION = "What did Acme order, and when is delivery due?"
SUBQUERY = "When is Acme's delivery due?"
ORDER = "Acme ordered 40 units."
DELIVERY = "Acme's delivery is due on 28 April."


class FixedEmbeddings:
    def get_vector_size(self):
        return 3

    async def embed_text(self, texts):
        return [
            [0.0, 1.0, 0.0] if text in (SUBQUERY, DELIVERY) else [1.0, 0.0, 0.0] for text in texts
        ]


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
                "cognee.modules.retrieval.hybrid_decomposition_retriever.LLMGateway.acreate_structured_output",
                AsyncMock(return_value=QueryDecomposition(subqueries=[SUBQUERY])),
            ),
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
        assert decomposed.decomposition_state.subqueries == [SUBQUERY]
    finally:
        await vector.close()
        await graph.close()
