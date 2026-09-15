"""All three features in one search on the combined branch: decomposition (SDK-694),
document metadata lines (SDK-693) and structured evidence (SDK-698), through the public
consumer with mocked engines and a mocked decomposition model."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition
from cognee.modules.search.types import SearchType

out_mod = importlib.import_module("cognee.modules.search.methods.get_retriever_output")
RETRIEVER_MODULE = "cognee.modules.retrieval.hybrid_decomposition_retriever"

QUESTION = "How many units and when?"
LEG_A = "How many units?"
LEG_B = "When?"
VECTORS = {QUESTION: [1.0, 0.0, 0.0], LEG_A: [0.0, 1.0, 0.0], LEG_B: [0.0, 0.0, 1.0]}
STORED_METADATA = '{"created_at": "2024-01-15", "other": "hidden"}'


def _chunk(chunk_id, text, document):
    return SimpleNamespace(
        id=chunk_id,
        score=0.9,
        payload={
            "id": chunk_id,
            "text": text,
            "document_id": document,
            "document_name": f"{document}.txt",
            "chunk_index": 0,
            "external_metadata": STORED_METADATA,
        },
    )


def _engines():
    by_vector = {
        tuple(VECTORS[QUESTION]): [_chunk("c1", "Ordered 40 units.", "d1")],
        tuple(VECTORS[LEG_A]): [_chunk("c1", "Ordered 40 units.", "d1")],
        tuple(VECTORS[LEG_B]): [_chunk("c2", "Delivery 28 April.", "d2")],
    }

    async def search(collection_name, *args, **kwargs):
        if collection_name == "DocumentChunk_text":
            return list(by_vector.get(tuple(kwargs.get("query_vector") or ()), []))
        if collection_name == "Entity_name":
            return [SimpleNamespace(id="e1", score=0.9, payload={"id": "e1", "name": "Acme"})]
        return []

    vector = MagicMock()
    vector.search = AsyncMock(side_effect=search)
    vector.embedding_engine.embed_text = AsyncMock(
        side_effect=lambda texts: [VECTORS.get(text, [9.0, 9.0, 9.0]) for text in texts]
    )
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_neighborhood = AsyncMock(
        return_value=(
            [("e1", {"name": "Acme"}), ("o1", {"name": "Osaka"})],
            [
                (
                    "e1",
                    "o1",
                    "ships_from",
                    {"edge_text": "Acme ships from Osaka.", "edge_object_id": "edge-1"},
                )
            ],
        )
    )
    return SimpleNamespace(vector=vector, graph=graph)


@pytest.mark.asyncio
async def test_metadata_evidence_and_decomposition_compose_in_one_search():
    decompose = AsyncMock(return_value=QueryDecomposition(subqueries=[LEG_A, LEG_B]))

    with (
        patch.object(
            out_mod,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(is_empty=AsyncMock(return_value=False)),
        ),
        patch(
            f"{RETRIEVER_MODULE}.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_engines(),
        ),
        patch(f"{RETRIEVER_MODULE}.LLMGateway.acreate_structured_output", decompose),
        patch.object(out_mod, "graph_source_evidence", new_callable=AsyncMock, return_value=[]),
    ):
        result = await out_mod.get_retriever_output(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION,
            QUESTION,
            only_context=True,
            include_references=True,
            dataset=SimpleNamespace(id=uuid4(), name="ds", tenant_id=uuid4()),
            retriever_specific_config={
                "text_summaries_top_k": 0,
                "include_external_metadata": True,
                "external_metadata_keys": ["created_at"],
            },
        )

    # SDK-694: the leg's passage is merged in next to pass 1.
    assert "Ordered 40 units." in result.context
    assert "Delivery 28 April." in result.context
    # SDK-693: only the allowlisted key renders, above each passage.
    assert "created_at: 2024-01-15\nOrdered 40 units." in result.context
    assert "created_at: 2024-01-15\nDelivery 28 April." in result.context
    assert "hidden" not in result.context
    # SDK-698: structured evidence is built over the merged objects.
    kinds = {(reference.kind, reference.artifact_id) for reference in result.evidence}
    assert {
        ("segment", "c1"),
        ("segment", "c2"),
        ("graph_node", "e1"),
        ("graph_edge", "edge-1"),
    } <= kinds
    assert result.search_type is SearchType.HYBRID_COMPLETION_DECOMPOSITION
