import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.retrieval.insights_retriever import InsightsRetriever
from cognee.tests.tasks.descriptive_metrics.metrics_test_utils import create_connected_test_graph
from cognee.infrastructure.databases.graph.get_graph_engine import create_graph_engine
import unittest
from cognee.infrastructure.databases.graph import get_graph_engine


class TestInsightsRetriever:
    @pytest.fixture
    def mock_retriever(self):
        return InsightsRetriever()

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.get_graph_engine")
    async def test_get_context_with_existing_node(self, mock_get_graph_engine, mock_retriever):
        """Test get_context when node exists in graph."""
        mock_graph = AsyncMock()
        mock_get_graph_engine.return_value = mock_graph

        # Mock graph response
        mock_graph.extract_node.return_value = {"id": "123"}
        mock_graph.get_connections.return_value = [
            ({"id": "123"}, {"relationship_name": "linked_to"}, {"id": "456"})
        ]

        result = await mock_retriever.get_context("123")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0][0]["id"] == "123"
        assert result[0][1]["relationship_name"] == "linked_to"
        assert result[0][2]["id"] == "456"
        mock_graph.extract_node.assert_called_once_with("123")
        mock_graph.get_connections.assert_called_once_with("123")

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.get_vector_engine")
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

    @pytest.mark.asyncio
    @patch("cognee.modules.retrieval.insights_retriever.get_graph_engine")
    @patch("cognee.modules.retrieval.insights_retriever.get_vector_engine")
    async def test_get_context_with_no_exact_node(
        self, mock_get_vector_engine, mock_get_graph_engine, mock_retriever
    ):
        """Test get_context when node does not exist in the graph and vector search is used."""
        mock_graph = AsyncMock()
        mock_get_graph_engine.return_value = mock_graph
        mock_graph.extract_node.return_value = None  # Node does not exist

        mock_vector = AsyncMock()
        mock_get_vector_engine.return_value = mock_vector

        mock_vector.search.side_effect = [
            [AsyncMock(id="vec_1", score=0.4)],  # Entity_name search
            [AsyncMock(id="vec_2", score=0.3)],  # EntityType_name search
        ]

        mock_graph.get_connections.side_effect = lambda node_id: [
            ({"id": node_id}, {"relationship_name": "related_to"}, {"id": "456"})
        ]

        result = await mock_retriever.get_context("non_existing_query")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0][0]["id"] == "vec_1"
        assert result[0][1]["relationship_name"] == "related_to"
        assert result[0][2]["id"] == "456"

        assert result[1][0]["id"] == "vec_2"
        assert result[1][1]["relationship_name"] == "related_to"
        assert result[1][2]["id"] == "456"

    @pytest.mark.asyncio
    async def test_get_context_with_none_query(self, mock_retriever):
        """Test get_context with a None query (should return empty list)."""
        result = await mock_retriever.get_context(None)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_completion_with_context(self, mock_retriever):
        """Test get_completion when context is already provided."""
        test_context = [({"id": "123"}, {"relationship_name": "linked_to"}, {"id": "456"})]
        result = await mock_retriever.get_completion("test_query", context=test_context)
        assert result == test_context
