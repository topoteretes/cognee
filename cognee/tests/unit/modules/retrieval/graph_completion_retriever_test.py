import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever


class TestGraphCompletionRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return GraphCompletionRetriever(system_prompt_path="test_prompt.txt")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_graph_engine")
    @patch("cognee.modules.retrieval.graph_completion_retriever.ChunksRetriever")
    async def test_get_completion(self, mock_chunks_retriever_class, mock_get_graph_engine, 
                                 mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query"
        doc_id1 = str(uuid.uuid4())
        doc_id2 = str(uuid.uuid4())
        
        # Mock chunks retriever
        mock_chunks_retriever = MagicMock()
        mock_chunks_retriever.get_completion = AsyncMock()
        mock_chunks_retriever.get_completion.return_value = [
            {
                "content": "This is the first chunk.",
                "document_id": doc_id1,
                "metadata": {"title": "Document 1"},
                "score": 0.95
            },
            {
                "content": "This is the second chunk.",
                "document_id": doc_id2,
                "metadata": {"title": "Document 2"},
                "score": 0.85
            }
        ]
        mock_chunks_retriever_class.return_value = mock_chunks_retriever
        
        # Mock graph engine
        mock_graph_engine = MagicMock()
        mock_graph_engine.get_graph_for_document = AsyncMock()
        mock_graph_engine.get_graph_for_document.return_value = {
            "nodes": [
                {"id": "node1", "label": "Node 1", "properties": {"name": "Node 1"}},
                {"id": "node2", "label": "Node 2", "properties": {"name": "Node 2"}}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "label": "RELATES_TO"}
            ]
        }
        mock_get_graph_engine.return_value = mock_graph_engine
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with context and graph"
        
        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock()
        mock_llm_client.complete.return_value = "Generated graph completion response"
        mock_get_llm_client.return_value = mock_llm_client
        
        # Execute
        results = await mock_retriever.get_completion(query)
        
        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "Generated graph completion response"
        
        # Verify chunks retriever was called
        mock_chunks_retriever.get_completion.assert_called_once_with(query)
        
        # Verify graph engine was called for both documents
        assert mock_graph_engine.get_graph_for_document.call_count == 2
        
        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()
        
        # Verify LLM client was called
        mock_llm_client.complete.assert_called_once_with("Rendered prompt with context and graph")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_graph_engine")
    @patch("cognee.modules.retrieval.graph_completion_retriever.ChunksRetriever")
    async def test_get_completion_with_empty_chunks(self, mock_chunks_retriever_class, mock_get_graph_engine, 
                                                  mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query with no chunks"
        
        # Mock chunks retriever with empty results
        mock_chunks_retriever = MagicMock()
        mock_chunks_retriever.get_completion = AsyncMock()
        mock_chunks_retriever.get_completion.return_value = []
        mock_chunks_retriever_class.return_value = mock_chunks_retriever
        
        # Mock graph engine
        mock_graph_engine = MagicMock()
        mock_get_graph_engine.return_value = mock_graph_engine
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with no context or graph"
        
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
        
        # Verify chunks retriever was called
        mock_chunks_retriever.get_completion.assert_called_once_with(query)
        
        # Verify graph engine was not called
        mock_graph_engine.get_graph_for_document.assert_not_called()
        
        # Verify prompt was rendered
        mock_render_prompt.assert_called_once()
        
        # Verify LLM client was called
        mock_llm_client.complete.assert_called_once_with("Rendered prompt with no context or graph")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_llm_client")
    @patch("cognee.modules.retrieval.graph_completion_retriever.render_prompt")
    @patch("cognee.modules.retrieval.graph_completion_retriever.get_graph_engine")
    @patch("cognee.modules.retrieval.graph_completion_retriever.ChunksRetriever")
    async def test_get_completion_with_empty_graph(self, mock_chunks_retriever_class, mock_get_graph_engine, 
                                                 mock_render_prompt, mock_get_llm_client, mock_retriever):
        # Setup
        query = "test query with empty graph"
        doc_id = str(uuid.uuid4())
        
        # Mock chunks retriever
        mock_chunks_retriever = MagicMock()
        mock_chunks_retriever.get_completion = AsyncMock()
        mock_chunks_retriever.get_completion.return_value = [
            {
                "content": "This is a chunk with no graph.",
                "document_id": doc_id,
                "metadata": {"title": "Document"},
                "score": 0.95
            }
        ]
        mock_chunks_retriever_class.return_value = mock_chunks_retriever
        
        # Mock graph engine with empty graph
        mock_graph_engine = MagicMock()
        mock_graph_engine.get_graph_for_document = AsyncMock()
        mock_graph_engine.get_graph_for_document.return_value = {
            "nodes": [],
            "edges": []
        }
        mock_get_graph_engine.return_value = mock_graph_engine
        
        # Mock render_prompt
        mock_render_prompt.return_value = "Rendered prompt with context but no graph"
        
        # Mock LLM client
        mock_llm_client = MagicMock()
        mock_llm_client.complete = AsyncMock()
        mock_llm_client.complete.return_value = "Response without graph context"
        mock_get_llm_client.return_value = mock_llm_client
        
        # Execute
        results = await mock_retriever.get_completion(query)
        
        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "Response without graph context"
        
        # Verify graph engine was called
        mock_graph_engine.get_graph_for_document.assert_called_once_with(doc_id)
        
        # Verify prompt was rendered
        mock_render_prompt.assert_called_once() 