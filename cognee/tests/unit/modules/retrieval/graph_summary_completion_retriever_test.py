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
    @patch("cognee.modules.retrieval.utils.completion.read_query_prompt")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_default_user")
    async def test_get_completion_with_custom_system_prompt(
        self,
        mock_get_default_user,
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

        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = (
            "Generated graph summary completion response"
        )
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        results = await mock_retriever.get_completion(query, context="test context")

        # Verify
        assert len(results) == 1

        # Verify render_prompt was called with custom prompt path
        mock_render_prompt.assert_called_once()
        assert mock_render_prompt.call_args[0][0] == "custom_user_prompt.txt"

        mock_read_query_prompt.assert_called_once()
        assert mock_read_query_prompt.call_args[0][0] == "custom_system_prompt.txt"

        mock_llm_client.acreate_structured_output.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "cognee.modules.retrieval.graph_completion_retriever.GraphCompletionRetriever.resolve_edges_to_text"
    )
    @patch(
        "cognee.modules.retrieval.graph_summary_completion_retriever.summarize_text",
        new_callable=AsyncMock,
    )
    async def test_resolve_edges_to_text_calls_super_and_summarizes(
        self, mock_summarize_text, mock_resolve_edges_to_text, mock_retriever
    ):
        """Test resolve_edges_to_text calls the parent method and summarizes the result."""

        mock_resolve_edges_to_text.return_value = "Raw graph edges text"
        mock_summarize_text.return_value = "Summarized graph text"

        result = await mock_retriever.resolve_edges_to_text(["mock_edge"])

        mock_resolve_edges_to_text.assert_called_once_with(["mock_edge"])
        mock_summarize_text.assert_called_once_with(
            "Raw graph edges text", mock_retriever.summarize_prompt_path
        )

        assert result == "Summarized graph text"
