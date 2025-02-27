import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_code_description_to_code_part_no_results():
    """Test that code_description_to_code_part handles no search results."""

    mock_user = AsyncMock()
    mock_user.id = "user123"
    mock_vector_engine = AsyncMock()
    mock_vector_engine.search.return_value = []

    with (
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_graph_engine",
            return_value=AsyncMock(),
        ),
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.CogneeGraph",
            return_value=AsyncMock(),
        ),
    ):
        from cognee.modules.retrieval.utils.description_to_codepart_search import (
            code_description_to_code_part,
        )

        result = await code_description_to_code_part("search query", mock_user, 2)

        assert result == []


@pytest.mark.asyncio
async def test_code_description_to_code_part_invalid_query():
    """Test that code_description_to_code_part raises ValueError for invalid query."""

    mock_user = AsyncMock()

    with pytest.raises(ValueError, match="The query must be a non-empty string."):
        from cognee.modules.retrieval.utils.description_to_codepart_search import (
            code_description_to_code_part,
        )

        await code_description_to_code_part("", mock_user, 2)


@pytest.mark.asyncio
async def test_code_description_to_code_part_invalid_top_k():
    """Test that code_description_to_code_part raises ValueError for invalid top_k."""

    mock_user = AsyncMock()

    with pytest.raises(ValueError, match="top_k must be a positive integer."):
        from cognee.modules.retrieval.utils.description_to_codepart_search import (
            code_description_to_code_part,
        )

        await code_description_to_code_part("search query", mock_user, 0)


@pytest.mark.asyncio
async def test_code_description_to_code_part_initialization_error():
    """Test that code_description_to_code_part raises RuntimeError for engine initialization errors."""

    mock_user = AsyncMock()

    with (
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_vector_engine",
            side_effect=Exception("Engine init failed"),
        ),
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_graph_engine",
            return_value=AsyncMock(),
        ),
    ):
        from cognee.modules.retrieval.utils.description_to_codepart_search import (
            code_description_to_code_part,
        )

        with pytest.raises(
            RuntimeError, match="System initialization error. Please try again later."
        ):
            await code_description_to_code_part("search query", mock_user, 2)


@pytest.mark.asyncio
async def test_code_description_to_code_part_execution_error():
    """Test that code_description_to_code_part raises RuntimeError for execution errors."""

    mock_user = AsyncMock()
    mock_user.id = "user123"
    mock_vector_engine = AsyncMock()
    mock_vector_engine.search.side_effect = Exception("Execution error")

    with (
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_vector_engine",
            return_value=mock_vector_engine,
        ),
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.get_graph_engine",
            return_value=AsyncMock(),
        ),
        patch(
            "cognee.modules.retrieval.utils.description_to_codepart_search.CogneeGraph",
            return_value=AsyncMock(),
        ),
    ):
        from cognee.modules.retrieval.utils.description_to_codepart_search import (
            code_description_to_code_part,
        )

        with pytest.raises(RuntimeError, match="An error occurred while processing your request."):
            await code_description_to_code_part("search query", mock_user, 2)
