import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_summary_completion_retriever import (
    GraphSummaryCompletionRetriever,
)


class TestGraphSummaryCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphSummaryCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.GraphCompletionRetriever")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text")
    async def test_get_completion(
        self,
        mock_summarize_text,
        mock_graph_completion_retriever_class,
        mock_render_prompt,
        mock_get_llm_client,
        mock_retriever,
    ):
        # Setup
        query = "test query"

        # Mock graph completion retriever
        mock_graph_completion_retriever = MagicMock()
        mock_graph_completion_retriever.resolve_edges_to_text = AsyncMock()
        mock_graph_completion_retriever.resolve_edges_to_text.return_value = (
            "Edges converted to a single string."
        )
        mock_graph_completion_retriever_class.return_value = mock_graph_completion_retriever

        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with summaries"

        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph summary completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0] == "Generated graph summary completion response"

        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()

        mock_summarize_text.assert_called_once()

        # Verify LLM client was called
        mock_llm_client.acreate_structured_output.assert_called_once()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.completion.read_query_prompt")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.GraphCompletionRetriever")
    async def test_get_completion_with_custom_system_prompt(
        self,
        mock_graph_completion_retriever_class,
        mock_render_prompt,
        mock_read_query_prompt,
        mock_get_llm_client,
        mock_retriever,
    ):
        # Setup
        query = "test query with custom prompt"

        # Set custom system prompt
        mock_retriever.user_prompt_path = "custom_user_prompt.txt"
        mock_retriever.system_prompt_path = "custom_system_prompt.txt"

        # Mock graph completion retriever
        mock_graph_completion_retriever = MagicMock()
        mock_graph_completion_retriever.resolve_edges_to_text = AsyncMock()
        mock_graph_completion_retriever.resolve_edges_to_text.return_value = (
            "Edges converted to a single string."
        )
        mock_graph_completion_retriever_class.return_value = mock_graph_completion_retriever

        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph summary completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1

        # Verify render_prompt was called with custom prompt path
        mock_render_prompt.assert_called_once()
        assert mock_render_prompt.call_args[0][0] == "custom_user_prompt.txt"

        # Called once in completion and once again in summary
        assert mock_read_query_prompt.call_count == 2
        assert mock_read_query_prompt.call_args[0][0] == "custom_system_prompt.txt"
