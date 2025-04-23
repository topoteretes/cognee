import pytest
from unittest.mock import AsyncMock, patch
from cognee.modules.users.models import User
from cognee.modules.retrieval.exceptions import CollectionDistancesNotFoundError
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_search,
    brute_force_triplet_search,
)


@pytest.mark.asyncio
@patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine")
async def test_brute_force_search_collection_not_found(mock_get_vector_engine):
    user = User(id="test_user")
    query = "test query"
    collections = ["nonexistent_collection"]
    top_k = 5
    mock_memory_fragment = AsyncMock()
    mock_vector_engine = AsyncMock()
    mock_vector_engine.get_distance_from_collection_elements.return_value = []
    mock_get_vector_engine.return_value = mock_vector_engine

    with pytest.raises(CollectionDistancesNotFoundError):
        await brute_force_search(
            query, user, top_k, collections=collections, memory_fragment=mock_memory_fragment
        )


@pytest.mark.asyncio
@patch("cognee.modules.retrieval.utils.brute_force_triplet_search.get_vector_engine")
async def test_brute_force_triplet_search_collection_not_found(mock_get_vector_engine):
    user = User(id="test_user")
    query = "test query"
    collections = ["nonexistent_collection"]
    top_k = 5
    mock_memory_fragment = AsyncMock()
    mock_vector_engine = AsyncMock()
    mock_vector_engine.get_distance_from_collection_elements.return_value = []
    mock_get_vector_engine.return_value = mock_vector_engine

    with pytest.raises(CollectionDistancesNotFoundError):
        await brute_force_triplet_search(
            query, user, top_k, collections=collections, memory_fragment=mock_memory_fragment
        )
