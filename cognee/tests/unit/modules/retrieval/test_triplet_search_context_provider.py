import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)
from cognee.infrastructure.engine import DataPoint


class MockEntity(DataPoint):
    name: str = None
    description: str = None
    text: str = None


@pytest.mark.asyncio
async def test_triplet_search_context_provider_alignment():
    # Construct entities:
    # 1. Valid entity (Alice)
    # 2. Invalid entity (empty/None fields)
    # 3. Valid entity (Bob)
    entity1 = MockEntity()
    entity1.name = "Alice"

    entity2 = MockEntity()  # Invalid (all None)

    entity3 = MockEntity()
    entity3.name = "Bob"

    entities = [entity1, entity2, entity3]

    provider = TripletSearchContextProvider()

    # Mock get_memory_fragment and brute_force_triplet_search
    with (
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.get_memory_fragment",
            new_callable=AsyncMock,
        ),
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.brute_force_triplet_search",
            new_callable=MagicMock,
        ) as mock_search,
    ):

        async def search_alice(*args, **kwargs):
            return ["triplet_alice"]

        async def search_bob(*args, **kwargs):
            return ["triplet_bob"]

        mock_search.side_effect = [search_alice(), search_bob()]

        # Mock _format_triplets to return formatted strings
        with patch.object(provider, "_format_triplets", new_callable=AsyncMock) as mock_format:
            mock_format.side_effect = lambda triplets, name: f"Formatted {name}"

            result = await provider.get_context(entities, "test_query")

            # Assertions:
            # - brute_force_triplet_search should have been called twice (once for Alice, once for Bob)
            assert mock_search.call_count == 2

            # - _format_triplets should be called with:
            #   1. ["triplet_alice"] and "Alice"
            #   2. ["triplet_bob"] and "Bob"
            mock_format.assert_any_call(["triplet_alice"], "Alice")
            mock_format.assert_any_call(["triplet_bob"], "Bob")

            # - The output string should contain formatted Alice and Bob
            assert "Formatted Alice" in result
            assert "Formatted Bob" in result
