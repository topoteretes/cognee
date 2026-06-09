from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.session.session_manager import SessionTurnPreparation
from cognee.modules.retrieval.agentic_retriever import AgenticRetriever
from cognee.modules.retrieval.graph_completion_decomposition_retriever import (
    GraphCompletionDecompositionRetriever,
)


@pytest.mark.asyncio
async def test_agentic_retriever_accepts_session_turn_kwargs_for_parent_path():
    turn_preparation = SessionTurnPreparation(effective_query="effective")
    retriever = AgenticRetriever(user=SimpleNamespace(id="user-1"), dataset_id=uuid4())

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever."
        "GraphCompletionRetriever.get_completion_from_context",
        new_callable=AsyncMock,
        return_value=["answer"],
    ) as mock_parent:
        result = await retriever.get_completion_from_context(
            query="raw",
            retrieved_objects=[],
            context="context",
            effective_query="effective",
            turn_preparation=turn_preparation,
        )

    assert result == ["answer"]
    assert mock_parent.await_args.kwargs["effective_query"] == "effective"
    assert mock_parent.await_args.kwargs["turn_preparation"] is turn_preparation


@pytest.mark.asyncio
async def test_decomposition_retriever_accepts_session_turn_kwargs_for_parent_path():
    turn_preparation = SessionTurnPreparation(effective_query="effective")
    retriever = GraphCompletionDecompositionRetriever()
    retriever._ensure_state = AsyncMock(return_value=SimpleNamespace(merged_edges=[]))

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever."
        "GraphCompletionRetriever.get_completion_from_context",
        new_callable=AsyncMock,
        return_value=["answer"],
    ) as mock_parent:
        result = await retriever.get_completion_from_context(
            query="raw",
            retrieved_objects=[],
            context="context",
            effective_query="effective",
            turn_preparation=turn_preparation,
        )

    assert result == ["answer"]
    assert mock_parent.await_args.kwargs["effective_query"] == "effective"
    assert mock_parent.await_args.kwargs["turn_preparation"] is turn_preparation
