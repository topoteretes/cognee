"""Resolution reads are shared within a search, never across searches or datasets."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.context_global_variables import current_dataset_id
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.retrieval.graph_completion_cot_retriever import GraphCompletionCotRetriever
from cognee.modules.retrieval.hybrid_retriever import HybridRetriever
from cognee.modules.retrieval.utils.resolved_context import (
    read_user_resolutions,
    user_resolution_scope,
)


@pytest.mark.asyncio
async def test_concurrent_reads_share_one_node_only_query_and_next_search_reads_again():
    graph = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=([], [])))
    async with user_resolution_scope():
        results = await asyncio.gather(*(read_user_resolutions(graph) for _ in range(3)))
        assert results == [[], [], []]
        graph.get_filtered_graph_data.assert_awaited_once_with(
            [{"type": ["FactConflict"]}], include_edges=False
        )
    graph.get_filtered_graph_data.return_value = (
        [("decision", {"resolution_source": "user", "selected_fact": "New decision"})],
        [],
    )
    async with user_resolution_scope():
        assert (await read_user_resolutions(graph))[0]["selected_fact"] == "New decision"
    assert graph.get_filtered_graph_data.await_count == 2


@pytest.mark.asyncio
async def test_resolution_cache_does_not_cross_dataset_contexts_or_keep_failed_reads():
    graph = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=([], [])))
    async with user_resolution_scope():
        token = current_dataset_id.set(uuid4())
        try:
            await read_user_resolutions(graph)
            current_dataset_id.set(uuid4())
            await read_user_resolutions(graph)
        finally:
            current_dataset_id.reset(token)
    assert graph.get_filtered_graph_data.await_count == 2
    graph.get_filtered_graph_data.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        async with user_resolution_scope():
            await read_user_resolutions(graph)
    graph.get_filtered_graph_data.side_effect = None
    async with user_resolution_scope():
        assert await read_user_resolutions(graph) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [False, True])
async def test_hybrid_completion_reads_once_across_lanes_and_global_context(monkeypatch, batch):
    import cognee.modules.retrieval.hybrid_retriever as module

    chunk = SimpleNamespace(id="chunk", payload={"id": "chunk", "text": "A passage"}, score=0.1)
    vector = SimpleNamespace(
        has_collection=AsyncMock(return_value=False),
        embedding_engine=SimpleNamespace(embed_text=AsyncMock(return_value=[[0.1, 0.2]])),
        search=AsyncMock(
            side_effect=lambda collection, *a, **kw: (
                [chunk] if collection == "DocumentChunk_text" else []
            )
        ),
    )
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(return_value=([], [])),
        get_neighborhood=AsyncMock(return_value=([], [])),
        is_empty=AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        module,
        "get_unified_engine",
        AsyncMock(return_value=SimpleNamespace(graph=graph, vector=vector)),
    )
    monkeypatch.setattr(module, "load_root_text", AsyncMock(return_value="Root summary"))
    monkeypatch.setattr(module, "search_top_global_context_summaries", AsyncMock(return_value=[]))
    retriever = HybridRetriever(text_summaries_top_k=0, include_global_context_index=True)
    monkeypatch.setattr(retriever, "_use_session_cache", lambda: False)
    monkeypatch.setattr(
        retriever, "get_completion_from_context", AsyncMock(return_value=["answer"])
    )
    kwargs = {"query_batch": ["first", "second"]} if batch else {"query": "first"}
    await retriever.get_completion(**kwargs)
    assert graph.get_filtered_graph_data.await_count == 1
    await retriever.get_completion(**kwargs)
    assert graph.get_filtered_graph_data.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "retriever_type", [GraphCompletionCotRetriever, GraphCompletionContextExtensionRetriever]
)
async def test_followup_rounds_share_one_resolution_read(monkeypatch, retriever_type):
    import importlib

    graph = SimpleNamespace(get_filtered_graph_data=AsyncMock(return_value=([], [])))
    retriever = (
        retriever_type(max_iter=2)
        if retriever_type is GraphCompletionCotRetriever
        else retriever_type(context_extension_rounds=2)
    )
    retriever._unified_engine = SimpleNamespace(graph=graph)
    monkeypatch.setattr(retriever, "_use_session_cache", lambda: False)
    calls = 0

    async def triplets(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [
            Edge(Node("subject"), Node(f"value-{calls}"), attributes={"edge_text": f"fact-{calls}"})
        ]

    monkeypatch.setattr(retriever, "get_triplets", AsyncMock(side_effect=triplets))
    if retriever_type is GraphCompletionCotRetriever:
        monkeypatch.setattr(retriever, "_generate_completions", AsyncMock())
        monkeypatch.setattr(retriever, "_run_cot_round", AsyncMock(return_value=["followup"]))
    else:
        module = importlib.import_module(retriever_type.__module__)
        monkeypatch.setattr(
            module, "generate_completion_batch", AsyncMock(return_value=["followup"])
        )
    await retriever.get_retrieved_objects(query="question")
    assert calls == 3
    assert graph.get_filtered_graph_data.await_count == 1
