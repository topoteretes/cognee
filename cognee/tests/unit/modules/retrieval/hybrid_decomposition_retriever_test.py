"""HybridDecompositionRetriever: pass 1, context-primed decomposition, per-subquery hybrid
retrieval and an id-keyed union, with mocked lanes and a mocked decomposition LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.exceptions.exceptions import NoDataError, QueryValidationError
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_decomposition_retriever import (
    HybridDecompositionRetriever,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition

MODULE = "cognee.modules.retrieval.hybrid_decomposition_retriever"
PLANNER = "cognee.modules.retrieval.utils.query_decomposition"
QUESTION = "How many units did Acme order, and when is delivery?"
LEG_A = "How many units did Acme order?"
LEG_B = "When is Acme's delivery?"

# Every query embeds to its own vector so the chunk lane can answer per query.
VECTORS = {QUESTION: [1.0, 0.0, 0.0], LEG_A: [0.0, 1.0, 0.0], LEG_B: [0.0, 0.0, 1.0]}
UNKNOWN_VECTOR = [9.0, 9.0, 9.0]


def _chunk(chunk_id, text):
    hit = MagicMock()
    hit.id = chunk_id
    hit.payload = {"id": chunk_id, "text": text}
    hit.score = 0.9
    return hit


def _lanes(chunks_by_query=None, graph_empty=False, failing_query=None):
    """A unified engine whose chunk lane answers per query; entity and fact lanes are empty."""
    by_vector = {tuple(VECTORS[q]): hits for q, hits in (chunks_by_query or {}).items()}
    vector = MagicMock()

    async def embed(texts):
        if failing_query in texts:
            raise RuntimeError(f"embedding failed for {failing_query}")
        return [VECTORS.get(text, UNKNOWN_VECTOR) for text in texts]

    vector.embedding_engine.embed_text = AsyncMock(side_effect=embed)

    async def search(collection_name, *args, **kwargs):
        if collection_name != "DocumentChunk_text":
            return []
        return list(by_vector.get(tuple(kwargs.get("query_vector") or ()), []))

    vector.has_collection = AsyncMock(return_value=False)
    vector.search = AsyncMock(side_effect=search)
    unified = MagicMock()
    unified.vector = vector
    unified.graph.is_empty = AsyncMock(return_value=graph_empty)
    unified.graph.get_neighborhood = AsyncMock(return_value=([], []))
    unified.graph.get_filtered_graph_data = AsyncMock(return_value=([], []))
    return unified


def _chunk_searches(unified) -> int:
    return sum(
        1 for call in unified.vector.search.await_args_list if call.args[0] == "DocumentChunk_text"
    )


def _decomposition(*subqueries):
    return AsyncMock(return_value=QueryDecomposition(subqueries=list(subqueries)))


def _ids(chunks):
    return [chunk.id for chunk in chunks]


def _run(unified, decompose=None):
    return (
        patch(
            "cognee.modules.retrieval.hybrid_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=unified,
        ),
        patch(f"{PLANNER}.LLMGateway.acreate_structured_output", decompose or _decomposition()),
    )


THREE_LEGS = {
    QUESTION: [_chunk("c1", "Acme ordered 40 units.")],
    LEG_A: [_chunk("c2", "Order: 40 units"), _chunk("c3", "Delivery 28 April")],
    LEG_B: [_chunk("c3", "Delivery 28 April"), _chunk("c4", "Night delivery")],
}


@pytest.mark.asyncio
async def test_subqueries_merge_with_pass_one_and_dedupe_by_id():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with engine, decompose as llm:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    # Round-robin by rank, pass 1 first; c3 came back for both subqueries and appears once.
    assert _ids(merged["chunks"]) == ["c1", "c2", "c3", "c4"]
    assert set(merged) == set(empty_hybrid_result())
    # The decomposition is context-primed: it sees the question and the pass-1 passages,
    # under the shipped prompt with the configured subquery budget.
    text_input = llm.await_args.kwargs["text_input"]
    assert QUESTION in text_input
    assert "Acme ordered 40 units." in text_input
    assert "1 to 7 focused retrieval subqueries" in llm.await_args.kwargs["system_prompt"]
    assert llm.await_args.kwargs["response_model"] is QueryDecomposition
    assert _chunk_searches(unified) == 3  # pass 1 + two subqueries


@pytest.mark.asyncio
async def test_completion_answers_the_original_question_over_the_merged_context():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with (
        engine,
        decompose,
        patch(
            "cognee.modules.retrieval.hybrid_retriever.generate_completion",
            new_callable=AsyncMock,
            return_value="answer",
        ) as complete,
    ):
        answers = await retriever.get_completion(query=QUESTION)

    assert answers == ["answer"]
    assert complete.await_args.kwargs["query"] == QUESTION
    assert complete.await_args.kwargs["context"] == (
        "## Relevant passages\nAcme ordered 40 units.\n---\nOrder: 40 units\n---\n"
        "Delivery 28 April\n---\nNight delivery"
    )


@pytest.mark.asyncio
async def test_decomposition_failure_falls_back_to_the_plain_hybrid_result():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=AsyncMock(side_effect=RuntimeError("LLM down")))

    with engine, decompose, patch(f"{PLANNER}.logger") as log:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    assert _ids(merged["chunks"]) == ["c1"]
    assert _chunk_searches(unified) == 1
    assert "falling back to original query" in log.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_missing_prompt_file_skips_the_llm_and_falls_back():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(
        text_summaries_top_k=0, decomposition_system_prompt_path="no_such_prompt.txt"
    )
    engine, decompose = _run(unified)

    with engine, decompose as llm, patch(f"{PLANNER}.logger") as log:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    llm.assert_not_awaited()
    assert _ids(merged["chunks"]) == ["c1"]
    assert "prompt not found" in log.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_inline_decomposition_prompt_wins_over_the_prompt_file():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(
        text_summaries_top_k=0,
        decomposition_system_prompt="INLINE PROMPT",
        decomposition_system_prompt_path="never_read.txt",
    )
    engine, decompose = _run(unified, decompose=_decomposition(LEG_A))

    with engine, decompose as llm, patch(f"{MODULE}.render_prompt") as render:
        await retriever.get_retrieved_objects(query=QUESTION)

    render.assert_not_called()
    assert llm.await_args.kwargs["system_prompt"] == "INLINE PROMPT"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_subqueries", [7, 3])
async def test_subqueries_are_capped_at_max_subqueries(max_subqueries):
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0, max_subqueries=max_subqueries)
    engine, decompose = _run(unified, decompose=_decomposition(*(f"q{i}" for i in range(9))))

    with engine, decompose as llm:
        await retriever.get_retrieved_objects(query=QUESTION)

    assert f"1 to {max_subqueries} focused" in llm.await_args.kwargs["system_prompt"]
    assert _chunk_searches(unified) == 1 + max_subqueries


@pytest.mark.asyncio
async def test_the_original_question_as_a_subquery_reuses_pass_one():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=_decomposition(QUESTION))

    with engine, decompose:
        await retriever.get_retrieved_objects(query=QUESTION)

    assert _chunk_searches(unified) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_query", [QUESTION, LEG_B])
async def test_a_retrieval_failure_propagates_for_pass_one_and_subqueries(failing_query):
    unified = _lanes(THREE_LEGS, failing_query=failing_query)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with engine, decompose as llm, pytest.raises(RuntimeError, match="embedding failed"):
        await retriever.get_retrieved_objects(query=QUESTION)

    # Pass 1 fails before planning; a subquery fails after it.
    assert llm.await_count == (0 if failing_query == QUESTION else 1)


@pytest.mark.asyncio
async def test_query_batch_is_rejected():
    with pytest.raises(QueryValidationError, match="single query"):
        await HybridDecompositionRetriever().get_retrieved_objects(query_batch=["a", "b"])


@pytest.mark.asyncio
async def test_empty_graph_raises_without_decomposing():
    unified = _lanes(graph_empty=True)
    engine, decompose = _run(unified)

    with engine, decompose as llm, pytest.raises(NoDataError):
        await HybridDecompositionRetriever().get_retrieved_objects(query=QUESTION)

    llm.assert_not_awaited()
    unified.vector.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_retrievals_share_the_parent_user_resolution_scope():
    from cognee.modules.retrieval.utils.resolved_context import read_user_resolutions

    unified = _lanes()

    async def retrieve(self, query):
        await read_user_resolutions(unified.graph)
        return empty_hybrid_result()

    engine, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))
    with engine, decompose, patch.object(HybridRetriever, "_retrieve_one", new=retrieve):
        await HybridDecompositionRetriever().get_retrieved_objects(query=QUESTION)

    unified.graph.get_filtered_graph_data.assert_awaited_once()


@pytest.mark.parametrize("value", [0, -1])
def test_max_subqueries_below_one_is_rejected(value):
    with pytest.raises(ValueError, match="max_subqueries"):
        HybridDecompositionRetriever(max_subqueries=value)
