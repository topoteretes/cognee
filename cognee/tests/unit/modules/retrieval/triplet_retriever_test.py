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
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", objects)
        await retriever.get_completion_from_context("test query", objects, context)

    assert context == "Alice knows Bob\nBob works at Tech Corp"
    mock_vector_engine.search.assert_awaited_once_with(
        "Triplet_text", "test query", limit=5, include_payload=True
    )


@pytest.mark.asyncio
async def test_get_objects_no_collection(mock_vector_engine):
    """Test that NoDataError is raised when Triplet_text collection doesn't exist."""
    mock_vector_engine.has_collection.return_value = False

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="create_triplet_embeddings"):
            await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_results(mock_vector_engine):
    """Test that empty string is returned when no triplets are found."""
    mock_vector_engine.search.return_value = []

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        context = await retriever.get_context_from_objects("test query", [])

    assert context == ""


@pytest.mark.asyncio
async def test_get_objects_collection_not_found_error(mock_vector_engine):
    """Test that CollectionNotFoundError is converted to NoDataError."""
    mock_vector_engine.has_collection.side_effect = CollectionNotFoundError("Collection not found")

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(NoDataError, match="No data found"):
            await retriever.get_retrieved_objects("test query")


@pytest.mark.asyncio
async def test_get_context_empty_payload_text(mock_vector_engine):
    """Test get_context handles missing text in payload."""
    mock_result = MagicMock()
    mock_result.payload = {}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        with pytest.raises(KeyError):
            objects = await retriever.get_retrieved_objects("test query")
            await retriever.get_context_from_objects("test query", retrieved_objects=objects)


@pytest.mark.asyncio
async def test_get_context_single_triplet(mock_vector_engine):
    """Test get_context with single triplet result."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Single triplet"}

    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with patch(
        "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
        return_value=mock_vector_engine,
    ):
        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)

    assert context == "Single triplet"


@pytest.mark.asyncio
async def test_init_defaults():
    """Test TripletRetriever initialization with defaults."""
    retriever = TripletRetriever()

    assert retriever.user_prompt_path == "context_for_question.txt"
    assert retriever.system_prompt_path == "answer_simple_question.txt"
    assert retriever.top_k == 5  # Default is 5
    assert retriever.system_prompt is None


@pytest.mark.asyncio
async def test_init_custom_params():
    """Test TripletRetriever initialization with custom parameters."""
    retriever = TripletRetriever(
        user_prompt_path="custom_user.txt",
        system_prompt_path="custom_system.txt",
        system_prompt="Custom prompt",
        top_k=10,
    )

    assert retriever.user_prompt_path == "custom_user.txt"
    assert retriever.system_prompt_path == "custom_system.txt"
    assert retriever.system_prompt == "Custom prompt"
    assert retriever.top_k == 10


@pytest.mark.asyncio
async def test_get_completion_without_context(mock_vector_engine):
    """Test get_completion retrieves context when not provided."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context("test query", None, None)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_provided_context(mock_vector_engine):
    """Test get_completion uses provided context."""
    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        completion = await retriever.get_completion_from_context(
            "test query", None, context="Provided context"
        )

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"


@pytest.mark.asyncio
async def test_get_completion_with_session(mock_vector_engine):
    """Test get_completion with session caching enabled."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever(session_id="test_session")

    mock_user = MagicMock()
    mock_user.id = "test-user-id"

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_conversation_history",
            return_value="Previous conversation",
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.summarize_text",
            return_value="Context summary",
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.save_conversation_history",
        ) as mock_save,
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.triplet_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = mock_user

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert completion[0] == "Generated answer"
    mock_save.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_completion_with_session_no_user_id(mock_vector_engine):
    """Test get_completion with session config but no user ID."""
    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever()

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value="Generated answer",
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
        patch("cognee.modules.retrieval.triplet_retriever.session_user") as mock_session_user,
    ):
        mock_config = MagicMock()
        mock_config.caching = True
        mock_cache_config.return_value = mock_config
        mock_session_user.get.return_value = None  # No user

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert isinstance(completion, list)
    assert len(completion) == 1


@pytest.mark.asyncio
async def test_get_completion_with_response_model(mock_vector_engine):
    """Test get_completion with custom response model."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        answer: str

    mock_result = MagicMock()
    mock_result.payload = {"text": "Test triplet"}
    mock_vector_engine.has_collection.return_value = True
    mock_vector_engine.search.return_value = [mock_result]

    retriever = TripletRetriever(response_model=TestModel)

    with (
        patch(
            "cognee.modules.retrieval.triplet_retriever.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.triplet_retriever.generate_completion",
            return_value=TestModel(answer="Test answer"),
        ),
        patch("cognee.modules.retrieval.triplet_retriever.CacheConfig") as mock_cache_config,
    ):
        mock_config = MagicMock()
        mock_config.caching = False
        mock_cache_config.return_value = mock_config

        objects = await retriever.get_retrieved_objects("test query")
        context = await retriever.get_context_from_objects("test query", retrieved_objects=objects)
        completion = await retriever.get_completion_from_context("test query", objects, context)

    assert isinstance(completion, list)
    assert len(completion) == 1
    assert isinstance(completion[0], TestModel)


@pytest.mark.asyncio
async def test_init_none_top_k():
    """Test TripletRetriever initialization with None top_k."""
    retriever = TripletRetriever(top_k=None)

    assert retriever.top_k == 5
