import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from types import SimpleNamespace

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
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert len(completion) == 2
    assert completion[0]["text"] == "S.R."
    assert completion[1]["text"] == "M.B."
    mock_vector_engine.search.assert_awaited_once_with(
        "TextSummary_text", "test query", limit=5, include_payload=True
    )


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
            await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_objects_empty_results(mock_vector_engine):
    """Test that empty list is returned when no summaries are found."""
    mock_vector_engine.search.return_value = []

    retriever = SummariesRetriever()

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")

    assert objects == []


@pytest.mark.asyncio
async def test_get_objects_top_k_limit(mock_vector_engine):
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
        objects = await retriever.get_retrieved_objects("test query")

    assert len(objects) == 3
    mock_vector_engine.search.assert_awaited_once_with(
        "TextSummary_text", "test query", limit=3, include_payload=True
    )


@pytest.mark.asyncio
async def test_get_context_with_objects(mock_vector_engine):
    """Test get_completion returns provided context."""
    retriever = SummariesRetriever()

    provided_context = {"text": "S.R."}
    sn = SimpleNamespace()
    sn.payload = provided_context
    completion = await retriever.get_context_from_objects("test query", retrieved_objects=[sn])

    assert completion == provided_context["text"]


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
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert len(completion) == 1
    assert completion[0]["text"] == "S.R."


@pytest.mark.asyncio
async def test_init_defaults():
    """Test SummariesRetriever initialization with defaults."""
    retriever = SummariesRetriever()

    assert retriever.top_k == 5


@pytest.mark.asyncio
async def test_init_custom_top_k():
    """Test SummariesRetriever initialization with custom top_k."""
    retriever = SummariesRetriever(top_k=10)

    assert retriever.top_k == 10


@pytest.mark.asyncio
async def test_get_objects_empty_payload(mock_vector_engine):
    """Test get_context handles empty payload."""
    mock_result = MagicMock()
    mock_result.payload = {}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = SummariesRetriever()

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")

    assert len(objects) == 1
    assert objects[0].payload == {}


@pytest.mark.asyncio
async def test_get_completion_with_session_id(mock_vector_engine):
    """Test get_completion with session_id parameter."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "S.R."}
    mock_vector_engine.search.return_value = [mock_result]

    retriever = SummariesRetriever(session_id="test_session")

    with patch(
        "cognee.modules.retrieval.summaries_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert len(completion) == 1
    assert completion[0]["text"] == "S.R."
