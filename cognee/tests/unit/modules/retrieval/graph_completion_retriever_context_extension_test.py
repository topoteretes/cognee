import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.graph_completion_context_extension_retriever import (
    GraphCompletionContextExtensionRetriever,
)
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge


@pytest.fixture
def mock_edge():
    """Create a mock edge."""
    edge = MagicMock(spec=Edge)
    return edge


@pytest.mark.asyncio
async def test_get_triplets_inherited(mock_edge):
    """Test that get_triplets is inherited from parent class."""
    retriever = GraphCompletionContextExtensionRetriever()

    with patch(
        "cognee.modules.retrieval.graph_completion_retriever.brute_force_triplet_search",
        return_value=[mock_edge],
    ):
        triplets = await retriever.get_triplets("test query")

    assert len(triplets) == 1
    assert triplets[0] == mock_edge


@pytest.mark.asyncio
async def test_init_defaults():
    """Test GraphCompletionContextExtensionRetriever initialization with defaults."""
    retriever = GraphCompletionContextExtensionRetriever()

    assert retriever.top_k == 5
    assert retriever.user_prompt_path == "graph_context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test GraphCompletionContextExtensionRetriever initialization with custom parameters."""
    retriever = GraphCompletionContextExtensionRetriever(
        top_k=10,
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
    )

    assert retriever.top_k == 10
    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
