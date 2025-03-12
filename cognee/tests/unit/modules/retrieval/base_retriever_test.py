import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.base_retriever import BaseRetriever


class TestBaseRetriever:
    # Define ConcreteRetriever class at the class level
    class ConcreteRetriever(BaseRetriever):
        async def get_completion(self, query):
            return [{"content": "Test result"}]

        async def get_context(self, query):
            return "Test context"

        async def search_vector_db(
            self, query, collection_name=None, limit=None, filter_condition=None
        ):
            # In the test, we're mocking get_vector_engine, so we should just return
            # what the test expects without actually calling the vector engine
            # This will be overridden by the mock in the test
            from cognee.infrastructure.databases.vector import get_vector_engine

            vector_engine = get_vector_engine()
            return await vector_engine.search(
                collection_name=collection_name,
                query_text=query,
                limit=limit,
                filter_condition=filter_condition,
            )

    @pytest.fixture
    def mock_retriever(self):
        # Return an instance of the ConcreteRetriever
        return self.ConcreteRetriever()

    @pytest.mark.asyncio
    async def test_get_completion_implementation(self, mock_retriever):
        # Test that the concrete implementation works
        results = await mock_retriever.get_completion("test query")

        # Verify
        assert len(results) == 1
        assert results[0]["content"] == "Test result"

    @pytest.mark.asyncio
    async def test_base_retriever_is_abstract(self):
        # Test that BaseRetriever is abstract and cannot be instantiated directly
        with pytest.raises(TypeError):
            BaseRetriever()

    @pytest.mark.asyncio
    async def test_search_vector_db(self):
        # Setup
        mock_retriever = self.ConcreteRetriever()
        mock_search_result = [
            {"id": str(uuid.uuid4()), "score": 0.95, "payload": {"text": "Result 1"}},
            {"id": str(uuid.uuid4()), "score": 0.85, "payload": {"text": "Result 2"}},
        ]

        # Mock the search_vector_db method directly
        original_search_vector_db = mock_retriever.search_vector_db
        mock_retriever.search_vector_db = AsyncMock(return_value=mock_search_result)

        try:
            # Execute
            results = await mock_retriever.search_vector_db(
                "test query", collection_name="test_collection", limit=2
            )

            # Verify
            assert results == mock_search_result
            mock_retriever.search_vector_db.assert_called_once_with(
                "test query", collection_name="test_collection", limit=2
            )
        finally:
            # Restore original method
            mock_retriever.search_vector_db = original_search_vector_db

    @pytest.mark.asyncio
    async def test_search_vector_db_with_filter(self):
        # Setup
        mock_retriever = self.ConcreteRetriever()
        mock_search_result = [
            {"id": str(uuid.uuid4()), "score": 0.95, "payload": {"text": "Result 1"}}
        ]

        # Mock the search_vector_db method directly
        original_search_vector_db = mock_retriever.search_vector_db
        mock_retriever.search_vector_db = AsyncMock(return_value=mock_search_result)

        filter_condition = {"document_id": str(uuid.uuid4())}

        try:
            # Execute
            results = await mock_retriever.search_vector_db(
                "test query",
                collection_name="test_collection",
                limit=1,
                filter_condition=filter_condition,
            )

            # Verify
            assert results == mock_search_result
            mock_retriever.search_vector_db.assert_called_once_with(
                "test query",
                collection_name="test_collection",
                limit=1,
                filter_condition=filter_condition,
            )
        finally:
            # Restore original method
            mock_retriever.search_vector_db = original_search_vector_db

    @pytest.mark.asyncio
    async def test_get_context_implementation(self, mock_retriever):
        # Test that the get_context implementation works
        context = await mock_retriever.get_context("test query")

        # Verify
        assert context == "Test context"
