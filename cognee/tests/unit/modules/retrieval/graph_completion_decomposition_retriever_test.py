import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.graph_completion_decomposition_retriever import (
    DecompositionMode,
    GraphCompletionDecompositionRetriever,
)
from cognee.modules.retrieval.utils.query_decomposition import (
    DecompositionRunState,
    QueryDecomposition,
    SubqueryRunState,
    normalize_subqueries,
)


def _make_unified_mock(graph_engine=None):
    if graph_engine is None:
        graph_engine = AsyncMock()
    unified = AsyncMock()
    unified.graph = graph_engine
    unified.vector = MagicMock()
    return unified


def _make_edge(
    *,
    edge_object_id=None,
    source_id="source",
    target_id="target",
    relationship_name="related_to",
    directed=True,
):
    edge = MagicMock(spec=Edge)
    node1 = MagicMock()
    node1.id = source_id
    node2 = MagicMock()
    node2.id = target_id
    edge.node1 = node1
    edge.node2 = node2
    edge.directed = directed
    edge.attributes = {"relationship_name": relationship_name}
    if edge_object_id is not None:
        edge.attributes["edge_object_id"] = edge_object_id
    return edge


@pytest.mark.asyncio
async def test_init_defaults():
    retriever = GraphCompletionDecompositionRetriever()
    assert retriever.decomposition_mode is DecompositionMode.ANSWER_PER_SUBQUERY


def test_init_invalid_mode_raises():
    with pytest.raises(ValueError, match="unsupported"):
        GraphCompletionDecompositionRetriever(decomposition_mode="unsupported")


def test_normalize_subqueries_strips_caps_and_falls_back():
    normalized = normalize_subqueries(
        "Original query",
        [
            "  First query  ",
            "",
            "first query",
            "Second query",
            "Third query",
            "Fourth query",
            "Fifth query",
            "Sixth query",
        ],
    )

    assert normalized == [
        "First query",
        "first query",
        "Second query",
        "Third query",
        "Fourth query",
    ]
    assert len(normalized) == 5

    fallback = normalize_subqueries(
        "Original query",
        ["", "   "],
    )
    assert fallback == ["Original query"]


@pytest.mark.asyncio
async def test_decompose_query_falls_back_to_original_query_on_failure():
    retriever = GraphCompletionDecompositionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.read_query_prompt",
            return_value="Decomposition prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM unavailable"),
        ),
    ):
        subqueries = await retriever._decompose_query("Original query")

    assert subqueries == ["Original query"]


@pytest.mark.asyncio
async def test_decompose_query_returns_normalized_subqueries_on_success():
    retriever = GraphCompletionDecompositionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.read_query_prompt",
            return_value="Decomposition prompt",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.LLMGateway.acreate_structured_output",
            new_callable=AsyncMock,
            return_value=QueryDecomposition(
                subqueries=["  First query  ", "", "First query", "Second query"]
            ),
        ),
    ):
        subqueries = await retriever._decompose_query("Original query")

    assert subqueries == ["First query", "Second query"]


@pytest.mark.asyncio
async def test_decompose_query_falls_back_to_original_query_when_prompt_missing():
    retriever = GraphCompletionDecompositionRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_decomposition_retriever.read_query_prompt",
        return_value="",
    ):
        subqueries = await retriever._decompose_query("Original query")

    assert subqueries == ["Original query"]


@pytest.mark.asyncio
async def test_get_retrieved_objects_rejects_query_batch():
    retriever = GraphCompletionDecompositionRetriever()

    with pytest.raises(QueryValidationError, match="single query"):
        await retriever.get_retrieved_objects(query_batch=["query 1", "query 2"])


@pytest.mark.asyncio
async def test_combined_mode_uses_get_triplets_batch_and_returns_flat_deduped_edges():
    retriever = GraphCompletionDecompositionRetriever(
        decomposition_mode=DecompositionMode.COMBINED_TRIPLETS_CONTEXT
    )
    edge1 = _make_edge(edge_object_id="edge-1", source_id="a", target_id="b")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="b", target_id="c")
    edge3 = _make_edge(edge_object_id="edge-3", source_id="c", target_id="d")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1, edge2], [edge2, edge3]],
        ) as mock_get_triplets_batch,
    ):
        result = await retriever.get_retrieved_objects(query="Original query")

    assert result == [edge1, edge2, edge3]
    mock_get_triplets_batch.assert_awaited_once_with(["subquery 1", "subquery 2"])


@pytest.mark.asyncio
async def test_combined_mode_final_completion_uses_original_query():
    retriever = GraphCompletionDecompositionRetriever(
        decomposition_mode=DecompositionMode.COMBINED_TRIPLETS_CONTEXT
    )
    edge1 = _make_edge(edge_object_id="edge-1", source_id="a", target_id="b")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="b", target_id="c")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1], [edge2]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            new_callable=AsyncMock,
            return_value="Merged context",
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            new_callable=AsyncMock,
            return_value="Final answer",
        ) as mock_generate_completion,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(query="Original query")
        context = await retriever.get_context_from_objects(
            query="Original query",
            retrieved_objects=objects,
        )
        completion = await retriever.get_completion_from_context(
            query="Original query",
            retrieved_objects=objects,
            context=context,
        )

    assert context == "Merged context"
    assert completion == ["Final answer"]
    assert mock_generate_completion.await_args.kwargs["query"] == "Original query"


@pytest.mark.asyncio
async def test_answer_per_subquery_mode_preserves_order_and_runs_subquery_completion():
    retriever = GraphCompletionDecompositionRetriever()
    edge1 = _make_edge(edge_object_id="edge-1", source_id="a", target_id="b")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="b", target_id="c")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1], [edge2]],
        ) as mock_get_triplets_batch,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            new_callable=AsyncMock,
            side_effect=["Context one", "Context two"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.generate_completion",
            new_callable=AsyncMock,
            side_effect=["Answer one", "Answer two"],
        ) as mock_generate_completion,
    ):
        objects = await retriever.get_retrieved_objects(query="Original query")
        context = await retriever.get_context_from_objects(
            query="Original query",
            retrieved_objects=objects,
        )

    assert objects == [edge1, edge2]
    assert "Subquery 1: subquery 1" in context
    assert "Subquery 2: subquery 2" in context
    assert context.index("subquery 1") < context.index("subquery 2")
    assert "Answer one" in context
    assert "Answer two" in context
    mock_get_triplets_batch.assert_awaited_once_with(["subquery 1", "subquery 2"])
    assert mock_generate_completion.await_args_list == [
        call(
            query="subquery 1",
            context="Context one",
            user_prompt_path=retriever.user_prompt_path,
            system_prompt_path=retriever.system_prompt_path,
            system_prompt=retriever.system_prompt,
            response_model=str,
        ),
        call(
            query="subquery 2",
            context="Context two",
            user_prompt_path=retriever.user_prompt_path,
            system_prompt_path=retriever.system_prompt_path,
            system_prompt=retriever.system_prompt,
            response_model=str,
        ),
    ]


@pytest.mark.asyncio
async def test_answer_per_subquery_final_completion_uses_original_query():
    retriever = GraphCompletionDecompositionRetriever()
    edge1 = _make_edge(edge_object_id="edge-1", source_id="a", target_id="b")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="b", target_id="c")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1], [edge2]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            new_callable=AsyncMock,
            side_effect=["Context one", "Context two"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.generate_completion",
            new_callable=AsyncMock,
            side_effect=["Answer one", "Answer two"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.generate_completion",
            new_callable=AsyncMock,
            return_value="Final answer",
        ) as mock_final_completion,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects(query="Original query")
        context = await retriever.get_context_from_objects(
            query="Original query",
            retrieved_objects=objects,
        )
        completion = await retriever.get_completion_from_context(
            query="Original query",
            retrieved_objects=objects,
            context=context,
        )

    assert completion == ["Final answer"]
    assert mock_final_completion.await_args.kwargs["query"] == "Original query"


@pytest.mark.asyncio
async def test_answer_per_subquery_uses_session_only_for_final_completion():
    retriever = GraphCompletionDecompositionRetriever(session_id="session-1")
    edge1 = _make_edge(edge_object_id="edge-1", source_id="node-1", target_id="node-2")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="node-2", target_id="node-3")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)
    mock_user = MagicMock()
    mock_user.id = "user-1"
    mock_session_manager = MagicMock()
    mock_session_manager.generate_completion_with_session = AsyncMock(return_value="Final answer")

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1], [edge2]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.resolve_edges_to_text",
            new_callable=AsyncMock,
            side_effect=["Context one", "Context two"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.generate_completion",
            new_callable=AsyncMock,
            side_effect=["Answer one", "Answer two"],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.get_session_manager",
            return_value=mock_session_manager,
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        objects = await retriever.get_retrieved_objects(query="Original query")
        context = await retriever.get_context_from_objects(
            query="Original query",
            retrieved_objects=objects,
        )
        assert mock_session_manager.generate_completion_with_session.await_count == 0

        completion = await retriever.get_completion_from_context(
            query="Original query",
            retrieved_objects=objects,
            context=context,
        )

    assert completion == ["Final answer"]
    mock_session_manager.generate_completion_with_session.assert_awaited_once()
    assert (
        mock_session_manager.generate_completion_with_session.call_args.kwargs["query"]
        == "Original query"
    )


@pytest.mark.asyncio
async def test_combined_mode_internal_batch_does_not_fail_with_session_cache():
    retriever = GraphCompletionDecompositionRetriever(
        decomposition_mode=DecompositionMode.COMBINED_TRIPLETS_CONTEXT
    )
    edge1 = _make_edge(edge_object_id="edge-1", source_id="a", target_id="b")
    edge2 = _make_edge(edge_object_id="edge-2", source_id="b", target_id="c")

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=False)
    mock_user = MagicMock()
    mock_user.id = "user-1"

    with (
        patch(
            "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
            new_callable=AsyncMock,
            return_value=_make_unified_mock(mock_graph_engine),
        ),
        patch.object(
            retriever,
            "_decompose_query",
            new_callable=AsyncMock,
            return_value=["subquery 1", "subquery 2"],
        ),
        patch.object(
            retriever,
            "get_triplets_batch",
            new_callable=AsyncMock,
            return_value=[[edge1], [edge2]],
        ),
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.CacheConfig"
        ) as mock_cache_config,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.session_user"
        ) as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        result = await retriever.get_retrieved_objects(query="Original query")

    assert result == [edge1, edge2]


@pytest.mark.asyncio
async def test_get_retrieved_objects_and_context_handle_empty_graph():
    retriever = GraphCompletionDecompositionRetriever()

    mock_graph_engine = AsyncMock()
    mock_graph_engine.is_empty = AsyncMock(return_value=True)

    with patch(
        "cognee.modules.retrieval.graph_completion_decomposition_retriever.get_unified_engine",
        new_callable=AsyncMock,
        return_value=_make_unified_mock(mock_graph_engine),
    ):
        objects = await retriever.get_retrieved_objects(query="Original query")
        context = await retriever.get_context_from_objects(
            query="Original query",
            retrieved_objects=objects,
        )

    assert objects == []
    assert context == ""


@pytest.mark.asyncio
async def test_ensure_state_returns_cached_state_for_matching_query():
    retriever = GraphCompletionDecompositionRetriever()
    retriever._decomposition_state = DecompositionRunState(
        original_query="Original query",
        subqueries=[SubqueryRunState(query="subquery 1")],
        final_context="Cached context",
    )

    state = await retriever._ensure_state("Original query")

    assert state is retriever._decomposition_state


@pytest.mark.asyncio
async def test_ensure_state_raises_for_missing_query():
    retriever = GraphCompletionDecompositionRetriever()

    with pytest.raises(QueryValidationError, match="non-empty query"):
        await retriever._ensure_state(None)


@pytest.mark.asyncio
async def test_ensure_state_raises_when_retrieval_does_not_initialize_state():
    retriever = GraphCompletionDecompositionRetriever()

    with patch.object(
        retriever,
        "get_retrieved_objects",
        new_callable=AsyncMock,
        return_value=[],
    ):
        with pytest.raises(QueryValidationError, match="Failed to initialize decomposition state"):
            await retriever._ensure_state("Original query")


@pytest.mark.asyncio
async def test_get_context_returns_cached_final_context():
    retriever = GraphCompletionDecompositionRetriever()
    retriever._decomposition_state = DecompositionRunState(
        original_query="Original query",
        subqueries=[SubqueryRunState(query="subquery 1")],
        merged_edges=[],
        final_context="Cached context",
    )

    context = await retriever.get_context_from_objects(query="Original query")

    assert context == "Cached context"


@pytest.mark.asyncio
async def test_get_completion_from_context_without_context_uses_state_objects():
    retriever = GraphCompletionDecompositionRetriever()
    state = DecompositionRunState(
        original_query="Original query",
        subqueries=[SubqueryRunState(query="subquery 1")],
        merged_edges=[_make_edge(edge_object_id="edge-1")],
    )

    with (
        patch.object(
            retriever,
            "_ensure_state",
            new_callable=AsyncMock,
            return_value=state,
        ),
        patch.object(
            retriever,
            "get_context_from_objects",
            new_callable=AsyncMock,
            return_value="Resolved context",
        ) as mock_get_context,
        patch(
            "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_completion_from_context",
            new_callable=AsyncMock,
            return_value=["Final answer"],
        ) as mock_parent_completion,
    ):
        completion = await retriever.get_completion_from_context(query="Original query")

    assert completion == ["Final answer"]
    mock_get_context.assert_awaited_once_with(
        query="Original query",
        retrieved_objects=state.merged_edges,
    )
    mock_parent_completion.assert_awaited_once_with(
        query="Original query",
        retrieved_objects=state.merged_edges,
        context="Resolved context",
    )
