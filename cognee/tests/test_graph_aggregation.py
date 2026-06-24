import pytest
from unittest.mock import AsyncMock, patch
from cognee.modules.retrieval.graph_aggregation_retriever import GraphAggregationRetriever

@pytest.mark.asyncio
async def test_graph_aggregation_retriever_comprehensive():
    # 1. Arrange: Mock data payloads matching the retrieval structures
    mock_nodes = [
        ("1", {"type": "Person", "name": "Alice"}),
        ("2", {"type": "Person", "name": "Bob"}),
        ("3", {"type": "Company", "name": "CogneeCorp"}),
    ]
    mock_edges = [
        ("1", "2", "KNOWS", {}),
        ("1", "3", "WORKS_AT", {}),
    ]
    mock_retrieved_objects = {
        "nodes": mock_nodes,
        "edges": mock_edges
    }

    retriever = GraphAggregationRetriever()

    # 2. Act & Assert: Safeguard against any internal litellm adapter client lookups
    with patch("cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client.get_llm_client") as mock_get_client:
        mock_client = AsyncMock()
        mock_client.acreate_structured_output = AsyncMock(return_value="Mocked Summary Response")
        mock_get_client.return_value = mock_client

        # Call your context formatter method
        context_data = await retriever.get_context_from_objects(
            query="Analyze graph aggregations",
            retrieved_objects=mock_retrieved_objects
        )
        
        # Verify your structural dictionary evaluations are 100% correct
        assert "node_types" in context_data
        assert context_data["node_types"]["Person"] == 2
        assert context_data["node_types"]["Company"] == 1
        assert len(context_data["edges"]) == 2