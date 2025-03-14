import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.summaries_retriever import SummariesRetriever


class TestSummariesRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return SummariesRetriever()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.summaries_retriever.get_vector_engine")
    async def test_get_completion(self, mock_get_vector_engine, mock_retriever):
        # Setup
        query = "test query"
        doc_id1 = str(uuid.uuid4())
        doc_id2 = str(uuid.uuid4())

        # Mock search results
        mock_result_1 = MagicMock()
        mock_result_1.payload = {
            "id": str(uuid.uuid4()),
            "score": 0.95,
            "payload": {
                "text": "This is the first summary.",
                "document_id": doc_id1,
                "metadata": {"title": "Document 1"},
            },
        }
        mock_result_2 = MagicMock()
        mock_result_2.payload = {
            "id": str(uuid.uuid4()),
            "score": 0.85,
            "payload": {
                "text": "This is the second summary.",
                "document_id": doc_id2,
                "metadata": {"title": "Document 2"},
            },
        }

        mock_search_results = [mock_result_1, mock_result_2]
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 2

        # Check first result
        assert results[0]["payload"]["text"] == "This is the first summary."
        assert results[0]["payload"]["document_id"] == doc_id1
        assert results[0]["payload"]["metadata"]["title"] == "Document 1"
        assert results[0]["score"] == 0.95

        # Check second result
        assert results[1]["payload"]["text"] == "This is the second summary."
        assert results[1]["payload"]["document_id"] == doc_id2
        assert results[1]["payload"]["metadata"]["title"] == "Document 2"
        assert results[1]["score"] == 0.85

        # Verify search was called correctly
        mock_vector_engine.search.assert_called_once_with("TextSummary_text", query, limit=5)

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.summaries_retriever.get_vector_engine")
    async def test_get_completion_with_empty_results(self, mock_get_vector_engine, mock_retriever):
        # Setup
        query = "test query with no results"
        mock_search_results = []
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 0
        mock_vector_engine.search.assert_called_once_with("TextSummary_text", query, limit=5)

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.summaries_retriever.get_vector_engine")
    async def test_get_completion_with_custom_limit(self, mock_get_vector_engine, mock_retriever):
        # Setup
        query = "test query with custom limit"
        doc_id = str(uuid.uuid4())

        # Mock search results
        mock_result = MagicMock()
        mock_result.payload = {
            "id": str(uuid.uuid4()),
            "score": 0.95,
            "payload": {
                "text": "This is a summary.",
                "document_id": doc_id,
                "metadata": {"title": "Document 1"},
            },
        }

        mock_search_results = [mock_result]
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        # Set custom limit
        mock_retriever.limit = 10

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0]["payload"]["text"] == "This is a summary."

        # Verify search was called with custom limit
        mock_vector_engine.search.assert_called_once_with("TextSummary_text", query, limit=10)
