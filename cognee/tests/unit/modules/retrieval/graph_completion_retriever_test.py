import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.exceptions import EntityNotFoundError
from cognee.tasks.completion.exceptions import NoRelevantDataFound


class TestGraphCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search")
    async def test_get_triplets_success(self, mock_brute_force_triplet_search, mock_retriever):
        mock_brute_force_triplet_search.return_value = [
            AsyncMock(
                node1=AsyncMock(attributes={"text": "Node A"}),
                attributes={"relationship_type": "connects"},
                node2=AsyncMock(attributes={"text": "Node B"}),
            )
        ]

        result = await mock_retriever.get_triplets("test query")

        assert isinstance(result, list)
        assert len(result) > 0
        assert result[0].attributes["relationship_type"] == "connects"
        mock_brute_force_triplet_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search")
    async def test_get_triplets_no_results(self, mock_brute_force_triplet_search, mock_retriever):
        mock_brute_force_triplet_search.return_value = []

        with pytest.raises(NoRelevantDataFound):
            await mock_retriever.get_triplets("test query")

    @pytest.mark.asyncio
    async def test_resolve_edges_to_text(self, mock_retriever):
        triplets = [
            AsyncMock(
                node1=AsyncMock(attributes={"text": "Node A"}),
                attributes={"relationship_type": "connects"},
                node2=AsyncMock(attributes={"text": "Node B"}),
            ),
            AsyncMock(
                node1=AsyncMock(attributes={"text": "Node X"}),
                attributes={"relationship_type": "links"},
                node2=AsyncMock(attributes={"text": "Node Y"}),
            ),
        ]

        result = await mock_retriever.resolve_edges_to_text(triplets)

        expected_output = "Node A -- connects -- Node B\n---\nNode X -- links -- Node Y"
        assert result == expected_output

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_triplets",
        new_callable=AsyncMock,
    )
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text",
        new_callable=AsyncMock,
    )
    async def test_get_context(self, mock_resolve_edges_to_text, mock_get_triplets, mock_retriever):
        """Test get_context calls get_triplets and resolve_edges_to_text."""
        mock_get_triplets.return_value = ["mock_triplet"]
        mock_resolve_edges_to_text.return_value = "Mock Context"

        result = await mock_retriever.get_context("test query")

        assert result == "Mock Context"
        mock_get_triplets.assert_called_once_with("test query")
        mock_resolve_edges_to_text.assert_called_once_with(["mock_triplet"])

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_context"
    )
    @patch("cognee.modules.retrieval.graph_completion_retriever.generate_completion")
    async def test_get_completion_without_context(
        self, mock_generate_completion, mock_get_context, mock_retriever
    ):
        """Test get_completion when no context is provided (calls get_context)."""
        mock_get_context.return_value = "Mock Context"
        mock_generate_completion.return_value = "Generated Completion"

        result = await mock_retriever.get_completion("test query")

        assert result == ["Generated Completion"]
        mock_get_context.assert_called_once_with("test query")
        mock_generate_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.get_context"
    )
    @patch("cognee.modules.retrieval.graph_completion_retriever.generate_completion")
    async def test_get_completion_with_context(
        self, mock_generate_completion, mock_get_context, mock_retriever
    ):
        """Test get_completion when context is provided (does not call get_context)."""
        mock_generate_completion.return_value = "Generated Completion"

        result = await mock_retriever.get_completion("test query", context="Provided Context")

        assert result == ["Generated Completion"]
        mock_get_context.assert_not_called()
        mock_generate_completion.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine")
    async def test_get_completion_with_empty_graph(
        self,
        mock_get_graph_engine,
        mock_get_llm_client,
        mock_retriever,
    ):
        # Setup
        query = "test query with empty graph"

        # Mock graph engine with empty graph
        mock_graph_engine = MagicMock()
        mock_graph_engine.get_graph_data = AsyncMock()
        mock_graph_engine.get_graph_data.return_value = ([], [])
        mock_get_graph_engine.return_value = mock_graph_engine

        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        with pytest.raises(EntityNotFoundError):
            await mock_retriever.get_completion(query)

        # Verify graph engine was called
        mock_graph_engine.get_graph_data.assert_called_once()
