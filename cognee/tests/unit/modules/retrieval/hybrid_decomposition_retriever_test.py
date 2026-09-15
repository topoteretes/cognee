"""HybridDecompositionRetriever: pass 1, context-primed decomposition, per-leg hybrid
retrieval and a content-keyed union, with mocked lanes and a mocked decomposition LLM."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_decomposition_retriever import (
    HybridDecompositionRetriever,
)
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition

MODULE = "cognee.modules.retrieval.hybrid_decomposition_retriever"
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


def _lanes(chunks_by_query: dict[str, list] | None = None, graph_empty: bool = False):
    """A unified engine whose chunk lane answers per query; entity and fact lanes are empty."""
    by_vector = {tuple(VECTORS[q]): hits for q, hits in (chunks_by_query or {}).items()}
    vector = MagicMock()
    vector.embedding_engine.embed_text = AsyncMock(
        side_effect=lambda texts: [VECTORS.get(text, UNKNOWN_VECTOR) for text in texts]
    )

    async def search(collection_name, *args, **kwargs):
        if collection_name != "DocumentChunk_text":
            return []
        return list(by_vector.get(tuple(kwargs.get("query_vector") or ()), []))

    vector.search = AsyncMock(side_effect=search)
    unified = MagicMock()
    unified.vector = vector
    unified.graph.is_empty = AsyncMock(return_value=graph_empty)
    unified.graph.get_neighborhood = AsyncMock(return_value=([], []))
    return unified


def _chunk_searches(unified) -> int:
    return sum(
        1 for call in unified.vector.search.await_args_list if call.args[0] == "DocumentChunk_text"
    )


def _decomposition(*subqueries):
    return AsyncMock(return_value=QueryDecomposition(subqueries=list(subqueries)))


def _ids(chunks):
    return [chunk.id for chunk in chunks]


def _run(unified, prompt="Decompose.", decompose=None):
    return (
        patch(f"{MODULE}.get_unified_engine", new_callable=AsyncMock, return_value=unified),
        patch(f"{MODULE}.read_query_prompt", return_value=prompt),
        patch(f"{MODULE}.LLMGateway.acreate_structured_output", decompose or _decomposition()),
    )


@pytest.mark.asyncio
async def test_legs_merge_with_pass_one_and_dedupe_by_content_id():
    unified = _lanes(
        {
            QUESTION: [_chunk("c1", "Acme ordered 40 units.")],
            LEG_A: [_chunk("c2", "Order: 40 units"), _chunk("c3", "Delivery 28 April")],
            LEG_B: [_chunk("c3", "Delivery 28 April"), _chunk("c4", "Night delivery")],
        }
    )
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, prompt, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with engine, prompt, decompose as llm:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    # Round-robin by rank, pass 1 first; c3 came back from both legs and appears once.
    assert _ids(merged["chunks"]) == ["c1", "c2", "c3", "c4"]
    assert set(merged) == set(empty_hybrid_result())
    # The decomposition is context-primed: it sees the question and the pass-1 passages.
    text_input = llm.await_args.kwargs["text_input"]
    assert QUESTION in text_input
    assert "Acme ordered 40 units." in text_input
    assert llm.await_args.kwargs["system_prompt"] == "Decompose."
    assert llm.await_args.kwargs["response_model"] is QueryDecomposition
    state = retriever._decomposition_state
    assert state.subqueries == [LEG_A, LEG_B]
    assert _ids(state.pass_one["chunks"]) == ["c1"]
    assert [_ids(leg["chunks"]) for leg in state.legs] == [["c2", "c3"], ["c3", "c4"]]
    # pass 1 + two legs
    assert _chunk_searches(unified) == 3


@pytest.mark.asyncio
async def test_decomposition_failure_falls_back_to_the_plain_hybrid_result():
    unified = _lanes({QUESTION: [_chunk("c1", "Acme ordered 40 units.")]})
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, prompt, decompose = _run(
        unified, decompose=AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    )

    with engine, prompt, decompose, patch(f"{MODULE}.logger") as log:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    assert _ids(merged["chunks"]) == ["c1"]
    assert retriever._decomposition_state.subqueries == [QUESTION]
    # The failure is logged as a warning, never raised.
    assert log.warning.call_count == 1
    assert "falling back to original query" in log.warning.call_args.args[0]


@pytest.mark.asyncio
async def test_missing_prompt_file_skips_the_llm_and_falls_back():
    unified = _lanes({QUESTION: [_chunk("c1", "Acme ordered 40 units.")]})
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, prompt, decompose = _run(unified, prompt=None)

    with engine, prompt, decompose as llm, patch(f"{MODULE}.logger") as log:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    llm.assert_not_awaited()
    assert _ids(merged["chunks"]) == ["c1"]
    assert retriever._decomposition_state.subqueries == [QUESTION]
    assert log.warning.call_count == 1
    assert "prompt not found" in log.warning.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_subqueries, expected_legs", [(7, 7), (3, 3)])
async def test_subqueries_are_capped_at_max_subqueries(max_subqueries, expected_legs):
    unified = _lanes({QUESTION: [_chunk("c1", "Acme ordered 40 units.")]})
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0, max_subqueries=max_subqueries)
    nine = [f"subquery {index}" for index in range(9)]
    engine, prompt, decompose = _run(unified, decompose=_decomposition(*nine))

    with engine, prompt, decompose:
        await retriever.get_retrieved_objects(query=QUESTION)

    state = retriever._decomposition_state
    assert state.subqueries == nine[:expected_legs]
    assert len(state.legs) == expected_legs
    assert _chunk_searches(unified) == 1 + expected_legs


@pytest.mark.asyncio
async def test_query_batch_is_rejected_everywhere():
    retriever = HybridDecompositionRetriever()

    with pytest.raises(QueryValidationError):
        await retriever.get_retrieved_objects(query_batch=["a", "b"])
    with pytest.raises(QueryValidationError):
        await retriever.get_context_from_objects(query_batch=["a", "b"])
    with pytest.raises(QueryValidationError):
        await retriever.get_completion_from_context(query_batch=["a", "b"])


@pytest.mark.asyncio
async def test_empty_graph_returns_the_empty_shape_without_decomposing():
    unified = _lanes(graph_empty=True)
    retriever = HybridDecompositionRetriever()
    engine, prompt, decompose = _run(unified)

    with engine, prompt, decompose as llm:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    assert merged == empty_hybrid_result()
    llm.assert_not_awaited()
    unified.vector.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_inline_decomposition_prompt_wins_over_the_prompt_file():
    unified = _lanes({QUESTION: [_chunk("c1", "Acme ordered 40 units.")]})
    retriever = HybridDecompositionRetriever(
        text_summaries_top_k=0,
        decomposition_system_prompt="INLINE PROMPT",
        decomposition_system_prompt_path="never_read.txt",
    )
    engine, prompt, decompose = _run(unified, decompose=_decomposition(LEG_A))

    with engine, prompt as read_prompt, decompose as llm:
        await retriever.get_retrieved_objects(query=QUESTION)

    read_prompt.assert_not_called()
    assert llm.await_args.kwargs["system_prompt"] == "INLINE PROMPT"


@pytest.mark.asyncio
async def test_context_and_completion_reuse_the_run_state_instead_of_retrieving_again():
    unified = _lanes(
        {
            QUESTION: [_chunk("c1", "Acme ordered 40 units.")],
            LEG_A: [_chunk("c2", "Order: 40 units")],
            LEG_B: [_chunk("c3", "Delivery 28 April")],
        }
    )
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, prompt, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with (
        engine,
        prompt,
        decompose,
        patch(
            "cognee.modules.retrieval.hybrid_retriever.generate_completion",
            new_callable=AsyncMock,
            return_value="answer",
        ) as complete,
    ):
        await retriever.get_retrieved_objects(query=QUESTION)
        searches_after_retrieval = _chunk_searches(unified)
        context = await retriever.get_context_from_objects(query=QUESTION)
        answers = await retriever.get_completion_from_context(query=QUESTION)

    assert _chunk_searches(unified) == searches_after_retrieval  # nothing retrieved again
    assert context == (
        "## Relevant passages\nAcme ordered 40 units.\n---\nOrder: 40 units\n---\nDelivery 28 April"
    )
    assert answers == ["answer"]
    # The final answer is generated for the original question over the merged context.
    assert complete.await_args.kwargs["query"] == QUESTION
    assert complete.await_args.kwargs["context"] == context


@pytest.mark.asyncio
async def test_context_for_a_new_question_triggers_a_fresh_run():
    unified = _lanes({QUESTION: [_chunk("c1", "Acme ordered 40 units.")]})
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, prompt, decompose = _run(unified)

    with engine, prompt, decompose:
        context = await retriever.get_context_from_objects(query=QUESTION)

    assert context == "## Relevant passages\nAcme ordered 40 units."
    assert retriever._decomposition_state.original_query == QUESTION


@pytest.mark.asyncio
async def test_merged_channel_ceilings_cap_the_union():
    unified = _lanes(
        {
            QUESTION: [_chunk("c1", "one")],
            LEG_A: [_chunk("c2", "two"), _chunk("c3", "three")],
            LEG_B: [_chunk("c4", "four")],
        }
    )
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0, merged_chunks_limit=2)
    engine, prompt, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with engine, prompt, decompose:
        merged = await retriever.get_retrieved_objects(query=QUESTION)

    assert _ids(merged["chunks"]) == ["c1", "c2"]  # pass 1 first, then round robin


def test_default_merged_limits_scale_with_the_number_of_legs():
    retriever = HybridDecompositionRetriever(chunks_top_k=5, entities_top_k=4, facts_top_k=3)

    assert retriever._merged_limits(leg_count=2) == {
        "chunks_limit": 15,
        "entities_limit": 12,
        "facts_limit": 9,
    }


def test_session_cache_path_is_never_taken():
    assert HybridDecompositionRetriever(session_id="session-1")._use_session_cache() is False


def test_constructor_normalises_its_own_parameters():
    retriever = HybridDecompositionRetriever(
        max_subqueries=0, decomposition_system_prompt="", decomposition_system_prompt_path=""
    )

    assert retriever.max_subqueries == 7
    assert retriever.decomposition_system_prompt is None
    assert retriever.decomposition_system_prompt_path == "hybrid_decomposition_system_prompt.txt"
