import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.search.methods.get_retriever_output import _count_retrieved_objects
from cognee.modules.search.types import SearchType


def test_count_retrieved_objects_counts_structured_lists():
    assert _count_retrieved_objects({"chunks": [1, 2], "entities": [3]}) == 3


def test_count_retrieved_objects_preserves_existing_shapes():
    assert _count_retrieved_objects(None) == 0
    assert _count_retrieved_objects(["a", "b"]) == 2
    assert _count_retrieved_objects({"triplets": []}) == 0
    assert _count_retrieved_objects({"metadata": "value"}) == 1
    assert _count_retrieved_objects("answer") == 1


@pytest.mark.asyncio
async def test_only_context_persist_trace_skips_completion(monkeypatch):
    retriever_output_module = importlib.import_module(
        "cognee.modules.search.methods.get_retriever_output"
    )

    graph_engine = MagicMock()
    graph_engine.is_empty = AsyncMock(return_value=False)
    monkeypatch.setattr(
        retriever_output_module,
        "get_graph_engine",
        AsyncMock(return_value=graph_engine),
    )
    monkeypatch.setattr(
        retriever_output_module,
        "update_node_access_timestamps",
        AsyncMock(),
    )

    retriever = MagicMock()
    retriever.get_retrieved_objects = AsyncMock(return_value=["edge"])
    retriever.get_context_from_objects = AsyncMock(return_value="resolved context")
    retriever.get_completion_from_context = AsyncMock(return_value=["answer"])
    retriever.persist_context_trace = AsyncMock(return_value="qa-id")
    monkeypatch.setattr(
        retriever_output_module,
        "get_search_type_retriever_instance",
        AsyncMock(return_value=retriever),
    )

    result = await retriever_output_module.get_retriever_output(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="test query",
        only_context=True,
        persist_trace=True,
    )

    assert result.context == "resolved context"
    assert result.completion is None
    retriever.get_completion_from_context.assert_not_awaited()
    retriever.persist_context_trace.assert_awaited_once_with(
        query="test query",
        retrieved_objects=["edge"],
        context="resolved context",
    )
