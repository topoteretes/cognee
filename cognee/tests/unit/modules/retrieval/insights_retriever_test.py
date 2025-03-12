import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.insights_retriever import InsightsRetriever


class TestInsightsRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return InsightsRetriever()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.BaseRetriever.search_vector_db")
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
                    "text": "This is the first insight.",
                    "document_id": doc_id1,
                    "metadata": {"title": "Document 1"},
                },
            },
            {
                "id": str(uuid.uuid4()),
                "score": 0.85,
                "payload": {
                    "text": "This is the second insight.",
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
        assert results[0]["content"] == "This is the first insight."
        assert results[0]["document_id"] == doc_id1
        assert results[0]["metadata"]["title"] == "Document 1"
        assert results[0]["score"] == 0.95

        # Check second result
        assert results[1]["content"] == "This is the second insight."
        assert results[1]["document_id"] == doc_id2
        assert results[1]["metadata"]["title"] == "Document 2"
        assert results[1]["score"] == 0.85

        # Verify search was called correctly
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="insights", limit=5, filter_condition=None
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion_with_empty_results(self, mock_search_vector_db, mock_retriever):
        # Setup
        query = "test query with no results"
        mock_search_vector_db.return_value = []

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 0
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="insights", limit=5, filter_condition=None
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion_with_filter(self, mock_search_vector_db, mock_retriever):
        # Setup
        query = "test query with filter"
        doc_id = str(uuid.uuid4())

        # Mock search results
        mock_search_results = [
            {
                "id": str(uuid.uuid4()),
                "score": 0.95,
                "payload": {
                    "text": "This is an insight with filter.",
                    "document_id": doc_id,
                    "metadata": {"title": "Document"},
                },
            }
        ]
        mock_search_vector_db.return_value = mock_search_results

        # Set filter
        filter_condition = {"document_id": doc_id}

        # Execute
        results = await mock_retriever.get_completion(query, filter_condition=filter_condition)

        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "This is an insight with filter."

        # Verify search was called with filter
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="insights", limit=5, filter_condition=filter_condition
        )

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.BaseRetriever.search_vector_db")
    async def test_get_completion_with_custom_collection(
        self, mock_search_vector_db, mock_retriever
    ):
        # Setup
        query = "test query with custom collection"
        doc_id = str(uuid.uuid4())

        # Mock search results
        mock_search_results = [
            {
                "id": str(uuid.uuid4()),
                "score": 0.95,
                "payload": {
                    "text": "This is an insight from custom collection.",
                    "document_id": doc_id,
                    "metadata": {"title": "Document"},
                },
            }
        ]
        mock_search_vector_db.return_value = mock_search_results

        # Set custom collection
        mock_retriever.collection_name = "custom_insights"

        # Execute
        results = await mock_retriever.get_completion(query)

        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "This is an insight from custom collection."

        # Verify search was called with custom collection
        mock_search_vector_db.assert_called_once_with(
            query, collection_name="custom_insights", limit=5, filter_condition=None
        )
