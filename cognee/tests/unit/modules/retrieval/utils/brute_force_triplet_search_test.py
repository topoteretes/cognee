import pytest
from cognee.modules.retrieval.exceptions import CollectionDistancesNotFoundError
from cognee.modules.users.models import User
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_search,
    brute_force_triplet_search,
)


@pytest.mark.asyncio
async def test_brute_force_search_collection_not_found():
    user = User(id="test_user")
    query = "test query"
    collections = ["nonexistent_collection"]
    top_k = 5

    with pytest.raises(Exception) as exc_info:
        await brute_force_search(query, user, top_k, collections=collections)

    assert isinstance(exc_info.value.__cause__, CollectionDistancesNotFoundError)


@pytest.mark.asyncio
async def test_brute_force_triplet_search_collection_not_found():
    user = User(id="test_user")
    query = "test query"
    collections = ["nonexistent_collection"]
    top_k = 5

    with pytest.raises(Exception) as exc_info:
        await brute_force_triplet_search(query, user, top_k, collections=collections)

    assert isinstance(exc_info.value.__cause__, CollectionDistancesNotFoundError)
