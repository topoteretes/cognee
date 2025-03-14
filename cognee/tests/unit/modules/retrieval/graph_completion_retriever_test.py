import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.graph.exceptions import EntityNotFoundError


class TestGraphCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine")
    async def test_get_completion(
        self,
        mock_get_graph_engine,
        mock_render_prompt,
        mock_get_llm_client,
        mock_retriever,
    ):
        # Setup
        query = "test query"

        # Mock graph engine
        mock_graph_engine = MagicMock()
        mock_graph_engine.get_graph_data = AsyncMock()
        nodes = [
            {"id": "node1", "label": "Node 1", "properties": {"name": "Node 1"}},
            {"id": "node2", "label": "Node 2", "properties": {"name": "Node 2"}},
        ]
        nodes_data = [(node["id"], node) for node in nodes]
        edges = [{"source": "node1", "target": "node2", "label": "RELATES_TO"}]
        edges_data = [(edge["source"], edge["target"], edge["label"], edge) for edge in edges]
        mock_graph_engine.get_graph_data.return_value = (nodes_data, edges_data)
        mock_get_graph_engine.return_value = mock_graph_engine

        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with context and graph"

        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0] == "Generated graph completion response"

        assert mock_graph_engine.get_graph_data.call_count == 1

        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()

        # Verify LLM client was called
        mock_llm_client.acreate_structured_output.assert_called_once_with(
            text_input="Rendered prompt with context and graph",
            system_prompt=None,
            response_model=str,
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_graph_engine")
    async def test_get_completion_with_empty_graph(
        self,
        mock_get_graph_engine,
        mock_render_prompt,
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

        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with context but no graph"

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

        # Verify prompt was not rendered
        mock_render_prompt.assert_not_called()
