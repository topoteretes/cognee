"""HYBRID_COMPLETION_DECOMPOSITION through the public consumer, with the real factory and
retriever and only the engines and the LLM mocked: the recall(only_context=True) path."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition
from cognee.modules.search.methods.get_retriever_output import get_retriever_output
from cognee.modules.search.types import SearchType

get_retriever_output_module = importlib.import_module(
    "cognee.modules.search.methods.get_retriever_output"
)
PLANNER = "cognee.modules.retrieval.utils.query_decomposition"

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

    vector.has_collection = AsyncMock(return_value=False)
    vector.search = AsyncMock(side_effect=search)
    graph = MagicMock()
    graph.is_empty = AsyncMock(return_value=False)
    graph.get_neighborhood = AsyncMock(return_value=([], []))
    graph.get_filtered_graph_data = AsyncMock(return_value=([], []))
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
        patch.object(
            get_retriever_output_module,
            "hybrid_deferral_reason",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=unified,
        ),
        patch(f"{PLANNER}.LLMGateway.acreate_structured_output", decompose),
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
async def test_decomposed_context_contains_every_chunk_plain_hybrid_returns():
    """AC1, literally: the same question through HYBRID_COMPLETION and through the
    decomposition type on the same engines; every plain chunk id is in the merged result."""
    decompose = AsyncMock(return_value=QueryDecomposition(subqueries=[LEG_A, LEG_B]))
    config = {"text_summaries_top_k": 0}
    graph_engine = SimpleNamespace(is_empty=AsyncMock(return_value=False))

    with (
        patch.object(
            get_retriever_output_module,
            "get_graph_engine",
            new_callable=AsyncMock,
            return_value=graph_engine,
        ),
        patch.object(
            get_retriever_output_module,
            "hybrid_deferral_reason",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_unified(),
        ),
        patch(f"{PLANNER}.LLMGateway.acreate_structured_output", decompose),
    ):
        plain = await get_retriever_output(
            SearchType.HYBRID_COMPLETION,
            QUESTION,
            only_context=True,
            retriever_specific_config=config,
        )
        decomposed = await get_retriever_output(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION,
            QUESTION,
            only_context=True,
            retriever_specific_config=config,
        )

    plain_ids = {chunk.id for chunk in plain.result_object["chunks"]}
    decomposed_ids = {chunk.id for chunk in decomposed.result_object["chunks"]}
    assert plain_ids == {"c1"}
    assert plain_ids <= decomposed_ids
    assert decomposed_ids == {"c1", "c2", "c3"}
    assert plain.context in decomposed.context or plain.context.split("\n")[1] in decomposed.context


@pytest.mark.asyncio
@pytest.mark.parametrize("config", [{"wide_search_top_k": 100}, {"triplet_distance_penalty": 6.5}])
async def test_graph_only_knobs_are_rejected_like_plain_hybrid(config):
    from cognee.exceptions import CogneeValidationError

    with pytest.raises(CogneeValidationError):
        await get_retriever_output_module._effective_search_type(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION, QUESTION, config, False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"node_type": str}, "node_type=str is not NodeSet"),
        ({"neighborhood_depth": 2}, "neighborhood_depth is set"),
        ({"feedback_influence": 0.2}, "feedback_influence > 0"),
    ],
)
async def test_graph_only_requests_are_refused_with_the_reason(config, reason):
    """The type is always chosen explicitly, so it is never answered by another algorithm."""
    with pytest.raises(QueryValidationError) as refusal:
        await get_retriever_output_module._effective_search_type(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION, QUESTION, config, False
        )

    assert reason in refusal.value.message
    assert "Use GRAPH_COMPLETION_DECOMPOSITION" in refusal.value.message


@pytest.mark.asyncio
async def test_a_dataset_without_chunk_collections_is_refused():
    vector = SimpleNamespace(has_collection=AsyncMock(return_value=False))
    with (
        patch(
            "cognee.modules.search.methods.hybrid_deferral.get_vector_engine_async",
            new_callable=AsyncMock,
            return_value=vector,
        ),
        pytest.raises(QueryValidationError) as refusal,
    ):
        await get_retriever_output_module._effective_search_type(
            SearchType.HYBRID_COMPLETION_DECOMPOSITION, QUESTION, {}, False
        )

    assert "collections missing" in refusal.value.message
    assert "Use GRAPH_COMPLETION_DECOMPOSITION" in refusal.value.message


@pytest.mark.asyncio
async def test_plain_hybrid_still_defers_on_the_same_request():
    resolved = await get_retriever_output_module._effective_search_type(
        SearchType.HYBRID_COMPLETION, QUESTION, {"neighborhood_depth": 2}, False
    )

    assert resolved is SearchType.GRAPH_COMPLETION
