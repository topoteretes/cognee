import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.triplet_retriever import TripletRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.has_collection = AsyncMock(return_value=True)
    engine.search = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_context_success(mock_vector_engine):
    """Test successful retrieval of triplet context."""
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "Alice knows Bob"}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "Bob works at Tech Corp"}

    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    retriever = TripletRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == "Alice knows Bob\nBob works at Tech Corp"
    mock_vector_engine.search.assert_awaited_once_with("Triplet_text", "test query", limit=5)


@pytest.mark.asyncio
async def test_get_context_no_collection(mock_vector_engine):
    """Test that NoDataError is raised when Triplet_text collection doesn't exist."""
    mock_vector_engine.has_collection.return_value = False

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="create_triplet_embeddings"):
            await retriever.get_context("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty string is returned when no triplets are found."""
    mock_vector_engine.search.return_value = []

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == ""


@pytest.mark.asyncio
async def test_get_context_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.has_collection.side_effect = CollectionNotFoundError("Collection not found")

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_context("test query")


@pytest.mark.asyncio
async def test_get_completion_without_session(mock_vector_engine):
    """Test get_completion without session caching."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Alice knows Bob"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ), patch(
        "cognee.modules.retrieval.triplet_retriever.generate_completion",
        return_value="Generated answer",
    ), patch(
        "cognee.modules.retrieval.triplet_retriever.CacheConfig"
    ) as mock_cache_config:
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query")

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_vector_engine):
    """Test get_completion with provided context."""
    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.generate_completion",
        return_value="Generated answer",
    ), patch(
        "cognee.modules.retrieval.triplet_retriever.CacheConfig"
    ) as mock_cache_config:
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query", context="Provided context")

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
