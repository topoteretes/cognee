import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.chunks_retriever import ChunksRetriever


class TestChunksRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return ChunksRetriever()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.get_vector_engine")
    async def test_get_completion(self, mock_get_vector_engine, mock_retriever):
        # Setup
        query = "test query"
        doc_id1 = str(uuid.uuid4())
        doc_id2 = str(uuid.uuid4())

        # Mock search results
        mock_result_1 = MagicMock()
        mock_result_1.payload = {
            "id": str(uuid.uuid4()),
            "text": "This is the first chunk result.",
            "document_id": doc_id1,
            "metadata": {"title": "Document 1"},
        }

        mock_result_2 = MagicMock()
        mock_result_2.payload = {
            "id": str(uuid.uuid4()),
            "text": "This is the second chunk result.",
            "document_id": doc_id2,
            "metadata": {"title": "Document 2"},
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
        assert results[0]["text"] == "This is the first chunk result."
        assert results[0]["document_id"] == doc_id1
        assert results[0]["metadata"]["title"] == "Document 1"

        # Check second result
        assert results[1]["text"] == "This is the second chunk result."
        assert results[1]["document_id"] == doc_id2
        assert results[1]["metadata"]["title"] == "Document 2"

        # Verify search was called correctly
        mock_vector_engine.search.assert_called_once_with("DocumentChunk_text", query, limit=5)

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.get_vector_engine")
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
        mock_vector_engine.search.assert_called_once_with("DocumentChunk_text", query, limit=5)

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.get_vector_engine")
    async def test_get_completion_with_missing_fields(self, mock_get_vector_engine, mock_retriever):
        # Setup
        query = "test query with incomplete data"

        # Mock search results
        mock_result_1 = MagicMock()
        mock_result_1.payload = {
            "id": str(uuid.uuid4()),
            "text": "This chunk has no document_id.",
            # Missing document_id and metadata
        }
        mock_result_2 = MagicMock()
        mock_result_2.payload = {
            "id": str(uuid.uuid4()),
            # Missing text
            "document_id": str(uuid.uuid4()),
            "metadata": {"title": "Document with missing text"},
        }

        mock_search_results = [mock_result_1, mock_result_2]
        mock_vector_engine = AsyncMock()
        mock_vector_engine.search.return_value = mock_search_results
        mock_get_vector_engine.return_value = mock_vector_engine

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 2

        # First result should have content but no document_id
        assert results[0]["text"] == "This chunk has no document_id."
        assert "document_id" not in results[0]
        assert "metadata" not in results[0]

        # Second result should have document_id and metadata but no content
        assert "text" not in results[1]
        assert "document_id" in results[1]
        assert results[1]["metadata"]["title"] == "Document with missing text"
