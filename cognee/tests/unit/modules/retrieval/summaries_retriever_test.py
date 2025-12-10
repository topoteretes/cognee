import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from cognee.modules.retrieval.summaries_retriever import SummariesRetriever
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
    """Test successful retrieval of summary context."""
    mock_result1 = MagicMock()
    mock_result1.payload = {"text": "S.R.", "made_from": "chunk1"}
    mock_result2 = MagicMock()
    mock_result2.payload = {"text": "M.B.", "made_from": "chunk2"}

    mock_vector_engine.search.return_value = [mock_result1, mock_result2]

    retriever = SummariesRetriever(top_k=5)

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert len(context) == 2
    assert context[0]["text"] == "S.R."
    assert context[1]["text"] == "M.B."
    mock_vector_engine.search.assert_awaited_once_with("TextSummary_text", "test query", limit=5)


@pytest.mark.asyncio
async def test_get_context_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.search.side_effect = CollectionNotFoundError("Collection not found")

    retriever = SummariesRetriever()

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_context("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty list is returned when no summaries are found."""
    mock_vector_engine.search.return_value = []

    retriever = SummariesRetriever()

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert context == []


@pytest.mark.asyncio
async def test_get_context_top_k_limit(mock_vector_engine):
    """Test that top_k parameter limits the number of results."""
    mock_results = [MagicMock() for _ in range(3)]
    for i, result in enumerate(mock_results):
        result.payload = {"text": f"Summary {i}"}

    mock_vector_engine.search.return_value = mock_results

    retriever = SummariesRetriever(top_k=3)

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context("test query")

    assert len(context) == 3
    mock_vector_engine.search.assert_awaited_once_with("TextSummary_text", "test query", limit=3)


@pytest.mark.asyncio
async def test_get_completion_with_context(mock_vector_engine):
    """Test get_completion returns provided context."""
    retriever = SummariesRetriever()

    provided_context = [{"text": "S.R."}, {"text": "M.B."}]
    completion = await retriever.get_completion("test query", context=provided_context)

    assert completion == provided_context


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_vector_engine):
    """Test get_completion retrieves context when not provided."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "S.R."}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = SummariesRetriever()

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        completion = await retriever.get_completion("test query")

    assert len(completion) == 1
    assert completion[0]["text"] == "S.R."
