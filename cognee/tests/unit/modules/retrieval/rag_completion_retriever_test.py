import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.completion_retriever import CompletionRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError


@pytest.fixture
def mock_vector_engine():
    """Create a mock vector engine."""
    engine = AsyncMock()
    engine.search = AsyncMock()
    return engine


@pytest.mark.asyncio
async def test_get_context_success(mock_vector_engine):
    """Test successful retrieval of context."""
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "Steve Rodger"}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "Mike Broski"}

    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    retriever = CompletionRetriever(top_k=2)

    with patch(
        "cognee.modules.retrieval.completion_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == "Steve Rodger\nMike Broski"
    mock_vector_engine.search.assert_awaited_once_with("DocumentChunk_text", "test query", limit=2)


@pytest.mark.asyncio
async def test_get_context_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.search.side_effect = CollectionNotFoundError("Collection not found")

    retriever = CompletionRetriever()

    with patch(
        "cognee.modules.retrieval.completion_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_context("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty string is returned when no chunks are found."""
    mock_vector_engine.search.return_value = []

    retriever = CompletionRetriever()

    with patch(
        "cognee.modules.retrieval.completion_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == ""


@pytest.mark.asyncio
async def test_get_context_top_k_limit(mock_vector_engine):
    """Test that top_k parameter limits the number of results."""
    mock_results = [MagicMock() for _ in range(2)]
    for i, result in enumerate(mock_results):
        result.payload = {"text": f"Chunk {i}"}

    mock_vector_engine.search.return_value = mock_results

    retriever = CompletionRetriever(top_k=2)

    with patch(
        "cognee.modules.retrieval.completion_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == "Chunk 0\nChunk 1"
    mock_vector_engine.search.assert_awaited_once_with("DocumentChunk_text", "test query", limit=2)


@pytest.mark.asyncio
async def test_get_context_single_chunk(mock_vector_engine):
    """Test get_context with single chunk result."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Single chunk text"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever()

    with patch(
        "cognee.modules.retrieval.completion_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == "Single chunk text"


@pytest.mark.asyncio
async def test_get_completion_without_session(mock_vector_engine):
    """Test get_completion without session caching."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Chunk text"}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
    ):
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
    retriever = CompletionRetriever()

    with (
        patch(
            "cognee.modules.retrieval.completion_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.completion_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion("test query", context="Provided context")

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
