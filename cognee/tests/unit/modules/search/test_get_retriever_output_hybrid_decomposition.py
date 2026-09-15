"""HYBRID_COMPLETION_DECOMPOSITION through the public consumer, with the real factory and
retriever and only the engines and the LLM mocked: the recall(only_context=True) path."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType

get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)
RETRIEVER_MODULE = "cognee.modules.retrieval.hybrid_decomposition_retriever"

QUESTION = "How many units did Acme order, and when is delivery?"
LEG_A = "How many units did Acme order?"
LEG_B = "When is Acme's delivery?"
VECTORS = {QUESTION: [1.0, 0.0, 0.0], LEG_A: [0.0, 1.0, 0.0], LEG_B: [0.0, 0.0, 1.0]}


def _chunk(chunk_id, text):
    return SimpleNamespace(id=chunk_id, score=0.9, payload={"id": chunk_id, "text": text})


def _unified():
    by_vector = {
        tuple(VECTORS[QUESTION]): [_chunk("c1", "Acme ordered 40 units.")],
        tuple(VECTORS[LEG_A]): [_chunk("c1", "Acme ordered 40 units."), _chunk("c2", "Order: 40")],
        tuple(VECTORS[LEG_B]): [_chunk("c3", "Delivery 28 April")],
    }
    vector = MagicMock()
    vector.embedding_engine.embed_text = AsyncMock(
        side_effect=lambda texts: [VECTORS.get(text, [9.0, 9.0, 9.0]) for text in texts]
    )

    async def search(collection_name, *args, **kwargs):
        if collection_name != "DocumentChunk_text":
            return []
        return list(by_vector.get(tuple(kwargs.get("query_vector") or ()), []))

    vector.search = AsyncMock(side_effect=search)
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_neighborhood = AsyncMock(return_value=([], []))
    return SimpleNamespace(vector=vector, graph=graph)


@pytest.mark.asyncio
async def test_hybrid_decomposition_only_context_returns_the_merged_context():
    """AC1 + AC4: the merged context holds every chunk plain hybrid would return, and the
    per-call decomposition prompt reaches the LLM verbatim through retriever_specific_config."""
    unified = _unified()
    decompose = AsyncMock(return_value=QueryDecomposition(subqueries=[LEG_A, LEG_B]))

    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(is_empty=AsyncMock(return_value=False)),
        ),
        patch(
            f"{RETRIEVER_MODULE}.get_unified_engine", new_callable=AsyncMock, return_value=unified
        ),
        patch(f"{RETRIEVER_MODULE}.LLMGateway.acreate_structured_output", decompose),
    ):
        result = await get_retriever_output(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION,
            QUESTION,
            only_context=True,
            retriever_specific_config={
                "decomposition_system_prompt": "Split into per-slot subqueries.",
                "text_summaries_top_k": 0,
            },
        )

    # The new type is served as itself, not deferred to GRAPH_COMPLETION.
    assert result.search_type is SearchType.HYBRID_COMPLETION_DECOMPOSITION
    assert result.only_context is True
    # Pass 1 (c1) always contributes and is not repeated when leg A returns it again. Round
    # robin by rank: rank 0 of every leg (c1, c1 dropped, c3) before rank 1 of leg A (c2).
    assert result.context == (
        "## Relevant passages\nAcme ordered 40 units.\n---\nDelivery 28 April\n---\nOrder: 40"
    )
    assert [chunk.id for chunk in result.result_object["chunks"]] == ["c1", "c3", "c2"]
    # AC4: the inline prompt from retriever_specific_config is exactly what the LLM received.
    assert decompose.await_args.kwargs["system_prompt"] == "Split into per-slot subqueries."
    assert QUESTION in decompose.await_args.kwargs["text_input"]
    assert "Acme ordered 40 units." in decompose.await_args.kwargs["text_input"]


@pytest.mark.asyncio
async def test_hybrid_decomposition_llm_failure_never_reaches_the_caller():
    """AC3: a failing decomposition degrades to the plain hybrid context, without raising."""
    unified = _unified()

    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(is_empty=AsyncMock(return_value=False)),
        ),
        patch(
            f"{RETRIEVER_MODULE}.get_unified_engine", new_callable=AsyncMock, return_value=unified
        ),
        patch(
            f"{RETRIEVER_MODULE}.LLMGateway.acreate_structured_output",
            AsyncMock(side_effect=RuntimeError("LLM unavailable")),
        ),
    ):
        result = await get_retriever_output(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION,
            QUESTION,
            only_context=True,
            retriever_specific_config={"text_summaries_top_k": 0},
        )

    assert result.context == "## Relevant passages\nAcme ordered 40 units."
    assert [chunk.id for chunk in result.result_object["chunks"]] == ["c1"]
