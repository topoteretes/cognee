import importlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from cognee.modules.search.types import SearchType


@pytest.fixture
def retriever_output_mod():
    return importlib.import_module("cognee.modules.search.methods.get_retriever_output")


@pytest.mark.asyncio
async def test_empty_graph_returns_empty_payload_without_llm_call(
    monkeypatch, retriever_output_mod
):
    """Empty knowledge graph must return an empty result without calling the completion LLM
    (regression for the phantom 'no context' answer)."""
    mock_engine = MagicMock()
    mock_engine.is_empty = AsyncMock(return_value=True)

    async def mock_get_graph_engine():
        return mock_engine

    monkeypatch.setattr(retriever_output_mod, "get_graph_engine", mock_get_graph_engine)
    monkeypatch.setattr(
        retriever_output_mod,
        "get_search_type_retriever_instance",
        AsyncMock(
            side_effect=AssertionError(
                "get_search_type_retriever_instance must not be called on an empty graph"
            )
        ),
    )

    result = await retriever_output_mod.get_retriever_output(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text="What database mistake did Jane make?",
    )

    assert result.result_object == []
    assert result.completion == []
    assert result.context is None
    assert result.search_type == SearchType.GRAPH_COMPLETION
