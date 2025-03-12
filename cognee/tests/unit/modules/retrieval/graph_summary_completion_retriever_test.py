import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_summary_completion_retriever import GraphSummaryCompletionRetriever


class TestGraphSummaryCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphSummaryCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.SummariesRetriever")
    async def test_get_completion(self, mock_summaries_retriever_class, mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query"
        doc_id1 = str(uuid.uuid4())
        doc_id2 = str(uuid.uuid4())
        
        # Mock summaries retriever
        mock_summaries_retriever = MagicMock()
        mock_summaries_retriever.get_completion = AsyncMock()
        mock_summaries_retriever.get_completion.return_value = [
            {
                "content": "This is the first summary.",
                "document_id": doc_id1,
                "metadata": {"title": "Document 1"},
                "score": 0.95
            },
            {
                "content": "This is the second summary.",
                "document_id": doc_id2,
                "metadata": {"title": "Document 2"},
                "score": 0.85
            }
        ]
        mock_summaries_retriever_class.return_value = mock_summaries_retriever
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with summaries"
        
        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock()
        mock_llm_client.complete.return_value = "Generated summary completion response"
        mock_get_llm_client.return_value = mock_llm_client
        
        # Execute
        results = await mock_retriever.get_completion(query)
        
        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "Generated summary completion response"
        
        # Verify summaries retriever was called
        mock_summaries_retriever.get_completion.assert_called_once_with(query)
        
        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()
        
        # Verify LLM client was called
        mock_llm_client.complete.assert_called_once_with("Rendered prompt with summaries")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.SummariesRetriever")
    async def test_get_completion_with_empty_summaries(self, mock_summaries_retriever_class, 
                                                     mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query with no summaries"
        
        # Mock summaries retriever with empty results
        mock_summaries_retriever = MagicMock()
        mock_summaries_retriever.get_completion = AsyncMock()
        mock_summaries_retriever.get_completion.return_value = []
        mock_summaries_retriever_class.return_value = mock_summaries_retriever
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with no summaries"
        
        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock()
        mock_llm_client.complete.return_value = "I don't have enough information to answer."
        mock_get_llm_client.return_value = mock_llm_client
        
        # Execute
        results = await mock_retriever.get_completion(query)
        
        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "I don't have enough information to answer."
        
        # Verify summaries retriever was called
        mock_summaries_retriever.get_completion.assert_called_once_with(query)
        
        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()
        
        # Verify LLM client was called
        mock_llm_client.complete.assert_called_once_with("Rendered prompt with no summaries")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_summary_completion_retriever.SummariesRetriever")
    async def test_get_completion_with_custom_system_prompt(self, mock_summaries_retriever_class, 
                                                          mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query with custom prompt"
        doc_id = str(uuid.uuid4())
        
        # Set custom system prompt
        mock_retriever.system_prompt_path = "custom_prompt.txt"
        
        # Mock summaries retriever
        mock_summaries_retriever = MagicMock()
        mock_summaries_retriever.get_completion = AsyncMock()
        mock_summaries_retriever.get_completion.return_value = [
            {
                "content": "This is a summary for custom prompt.",
                "document_id": doc_id,
                "metadata": {"title": "Document"},
                "score": 0.95
            }
        ]
        mock_summaries_retriever_class.return_value = mock_summaries_retriever
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered custom prompt with summaries"
        
        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock()
        mock_llm_client.complete.return_value = "Custom prompt completion response"
        mock_get_llm_client.return_value = mock_llm_client
        
        # Execute
        results = await mock_retriever.get_completion(query)
        
        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "Custom prompt completion response"
        
        # Verify render_prompt was called with custom prompt path
        mock_render_prompt.assert_called_once()
        assert mock_render_prompt.call_args[0][0] == "custom_prompt.txt" 