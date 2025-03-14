from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.completion_retriever import CompletionRetriever


class TestCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return CompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.utils.completion.get_llm_client")
    @patch("cognee.modules.retrieval.utils.completion.render_prompt")
    @patch("cognee.modules.retrieval.completion_retriever.get_vector_engine")
    async def test_get_completion(
        self, mock_get_vector_engine, mock_render_prompt, mock_get_llm_client, mock_retriever
    ):
        # Setup
        query = "test query"

        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with context"

        mock_search_results = [MagicMock()]
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.acreate_structured_output = AsyncMock()
        mock_llm_client.acreate_structured_output.return_value = "Generated completion response"
        mock_get_llm_client.return_value = mock_llm_client

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0] == "Generated completion response"

        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()

        # Verify LLM client was called
        mock_llm_client.acreate_structured_output.assert_called_once_with(
            text_input="Rendered prompt with context", system_prompt=None, response_model=str
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.completion_retriever.generate_completion")
    @patch("cognee.modules.retrieval.completion_retriever.get_vector_engine")
    async def test_get_completion_with_custom_prompt(
        self, mock_get_vector_engine, mock_generate_completion, mock_retriever
    ):
        # Setup
        query = "test query with custom prompt"

        mock_search_results = [MagicMock()]
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        mock_retriever.user_prompt_path = "custom_user_prompt.txt"
        mock_retriever.system_prompt_path = "custom_system_prompt.txt"

        mock_generate_completion.return_value = "Custom prompt completion response"

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0] == "Custom prompt completion response"

        assert mock_generate_completion.call_args[1]["user_prompt_path"] == "custom_user_prompt.txt"
        assert (
            mock_generate_completion.call_args[1]["system_prompt_path"]
            == "custom_system_prompt.txt"
        )
