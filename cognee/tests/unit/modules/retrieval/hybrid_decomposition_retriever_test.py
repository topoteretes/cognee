"""HybridDecompositionRetriever: pass 1, context-primed decomposition, per-subquery hybrid
retrieval and an id-keyed union, with mocked lanes and a mocked decomposition LLM."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.exceptions import PromptPathOutsideBaseDirectoryError
from cognee.modules.retrieval import session_aware_completion
from cognee.modules.retrieval.exceptions.exceptions import NoDataError, QueryValidationError
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid_decomposition_retriever import (
    MAX_SUBQUERIES,
    HybridDecompositionRetriever,
)
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.query_decomposition import QueryDecomposition
from cognee.modules.search.types import SearchType

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
async def test_a_planning_failure_propagates_instead_of_answering_with_plain_hybrid():
    unified = _lanes(THREE_LEGS)
    retriever = HybridDecompositionRetriever(text_summaries_top_k=0)
    engine, decompose = _run(unified, decompose=AsyncMock(side_effect=RuntimeError("LLM down")))

    with engine, decompose, pytest.raises(RuntimeError, match="LLM down"):
        await retriever.get_retrieved_objects(query=QUESTION)


def test_a_missing_prompt_file_is_a_configuration_error():
    with pytest.raises(QueryValidationError, match="no_such_prompt.txt"):
        HybridDecompositionRetriever(decomposition_system_prompt_path="no_such_prompt.txt")


@pytest.mark.parametrize(
    "path",
    [
        "../../../../pyproject.toml",
        "../prompts/../../../../../pyproject.toml",
        str(Path(__file__).resolve()),
    ],
)
def test_a_prompt_path_outside_the_prompt_directory_is_refused(path):
    with pytest.raises(PromptPathOutsideBaseDirectoryError):
        HybridDecompositionRetriever(decomposition_system_prompt_path=path)


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


@pytest.mark.parametrize("value", [0, -1, MAX_SUBQUERIES + 1, 10**9, "7", 7.0, 2.5, True, None])
def test_max_subqueries_outside_its_type_and_range_is_rejected(value):
    with pytest.raises(QueryValidationError, match="max_subqueries"):
        HybridDecompositionRetriever(max_subqueries=value)


@pytest.mark.parametrize("value", [1, MAX_SUBQUERIES])
def test_max_subqueries_at_its_bounds_is_accepted(value):
    assert HybridDecompositionRetriever(max_subqueries=value).max_subqueries == value


@pytest.mark.asyncio
async def test_a_session_turn_runs_sequentially_and_keeps_the_whole_union(monkeypatch):
    """Concurrent session mode is on and a session is available. Plain hybrid would take the
    concurrent runner, whose merge cuts each lane back to its top-k; this type takes the
    sequential runner, so the union of pass 1 and every subquery reaches the answer intact."""
    monkeypatch.setattr(
        session_aware_completion,
        "CacheConfig",
        lambda: SimpleNamespace(session_search_mode=session_aware_completion.CONCURRENT_MODE),
    )
    monkeypatch.setattr(
        session_aware_completion,
        "session_user",
        SimpleNamespace(get=lambda: SimpleNamespace(id=uuid4())),
    )
    monkeypatch.setattr(
        session_aware_completion,
        "get_session_manager",
        lambda: SimpleNamespace(is_session_available_for_completion=lambda user_id: True),
    )
    monkeypatch.setattr(session_aware_completion, "update_node_access_timestamps", AsyncMock())
    concurrent_runner = AsyncMock()
    monkeypatch.setattr(session_aware_completion, "run_concurrent_session_turn", concurrent_runner)

    options = {"chunks_top_k": 1, "text_summaries_top_k": 0, "session_id": "s1"}
    assert session_aware_completion.should_run_concurrent(HybridRetriever(**options))
    retriever = HybridDecompositionRetriever(**options)
    assert not session_aware_completion.should_run_concurrent(retriever)

    retriever.prepare_session_turn_for_retrieval = AsyncMock(
        return_value=SimpleNamespace(should_answer=True, effective_query=None)
    )
    answered = {}

    # A plain function: the runner inspects this method's signature, which Python 3.10
    # cannot do for an AsyncMock.
    async def complete(query=None, retrieved_objects=None, context=None):
        answered["retrieved_objects"] = retrieved_objects
        return ["answer"]

    retriever.get_completion_from_context = complete
    retriever.merge_retrieved_objects = MagicMock()
    unified = _lanes({query: hits[:1] for query, hits in THREE_LEGS.items()})
    engine, decompose = _run(unified, decompose=_decomposition(LEG_A, LEG_B))

    with engine, decompose:
        objects, context, completion = await session_aware_completion.run_session_aware_completion(
            retriever,
            raw_query=QUESTION,
            original_search_type=SearchType.HYBRID_COMPLETION_DECOMPOSITION,
        )

    concurrent_runner.assert_not_awaited()
    retriever.merge_retrieved_objects.assert_not_called()
    # Three chunks with chunks_top_k=1: one per retrieval, none cut by a session merge.
    assert _ids(objects["chunks"]) == ["c1", "c2", "c3"]
    assert all(text in context for text in ("Acme ordered 40 units.", "Order: 40 units"))
    assert completion == ["answer"]
    assert answered["retrieved_objects"] is objects
