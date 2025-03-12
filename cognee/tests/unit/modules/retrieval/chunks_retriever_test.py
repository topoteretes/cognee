import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.chunks_retriever import ChunksRetriever


class TestChunksRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return ChunksRetriever()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion(self, mock_search_vector_db, mock_retriever):
        # Setup
        query = "test query"
        doc_id1 = str(uuid.uuid4())
        doc_id2 = str(uuid.uuid4())

        # Mock search results
        mock_search_results = [
            {
                "id": str(uuid.uuid4()),
                "score": 0.95,
                "payload": {
                    "text": "This is the first chunk result.",
                    "document_id": doc_id1,
                    "metadata": {"title": "Document 1"},
                },
            },
            {
                "id": str(uuid.uuid4()),
                "score": 0.85,
                "payload": {
                    "text": "This is the second chunk result.",
                    "document_id": doc_id2,
                    "metadata": {"title": "Document 2"},
                },
            },
        ]
        mock_search_vector_db.return_value = mock_search_results

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 2

        # Check first result
        assert results[0]["content"] == "This is the first chunk result."
        assert results[0]["document_id"] == doc_id1
        assert results[0]["metadata"]["title"] == "Document 1"
        assert results[0]["score"] == 0.95

        # Check second result
        assert results[1]["content"] == "This is the second chunk result."
        assert results[1]["document_id"] == doc_id2
        assert results[1]["metadata"]["title"] == "Document 2"
        assert results[1]["score"] == 0.85

        # Verify search was called correctly
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="chunks", limit=5, filter_condition=None
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion_with_empty_results(self, mock_search_vector_db, mock_retriever):
        # Setup
        query = "test query with no results"
        mock_search_vector_db.return_value = []

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 0
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="chunks", limit=5, filter_condition=None
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.chunks_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion_with_missing_fields(self, mock_search_vector_db, mock_retriever):
        # Setup
        query = "test query with incomplete data"

        # Mock search results with missing fields
        mock_search_results = [
            {
                "id": str(uuid.uuid4()),
                "score": 0.95,
                "payload": {
                    "text": "This chunk has no document_id."
                    # Missing document_id and metadata
                },
            },
            {
                "id": str(uuid.uuid4()),
                "score": 0.85,
                "payload": {
                    # Missing text
                    "document_id": str(uuid.uuid4()),
                    "metadata": {"title": "Document with missing text"},
                },
            },
        ]
        mock_search_vector_db.return_value = mock_search_results

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 2

        # First result should have content but no document_id
        assert results[0]["content"] == "This chunk has no document_id."
        assert "document_id" not in results[0]
        assert "metadata" not in results[0]
        assert results[0]["score"] == 0.95

        # Second result should have document_id and metadata but no content
        assert "content" not in results[1]
        assert "document_id" in results[1]
        assert results[1]["metadata"]["title"] == "Document with missing text"
        assert results[1]["score"] == 0.85
