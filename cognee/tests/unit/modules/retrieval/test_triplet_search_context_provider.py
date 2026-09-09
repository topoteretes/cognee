from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


@pytest.mark.asyncio
async def test_get_context_pairs_results_with_matching_entities_when_some_have_empty_text():
    provider = TripletSearchContextProvider()

    entities = [
        SimpleNamespace(name="", description="", text=""),
        SimpleNamespace(name="", description="entity b"),
        SimpleNamespace(name="", description="entity c"),
    ]

    async def fake_triplet_search(*, query: str, **_kwargs):
        entity_text = query[: -len(" user query")]
        return [SimpleNamespace(label=f"triplet-for-{entity_text}")]

    with (
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.get_memory_fragment",
            new=AsyncMock(return_value=object()),
        ),
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.brute_force_triplet_search",
            side_effect=fake_triplet_search,
        ),
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.format_triplets",
            side_effect=lambda triplets: triplets[0].label,
        ),
    ):
        context = await provider.get_context(entities, "user query")

    assert "Context for entity b:" in context
    assert "Context for entity c:" in context
    assert "triplet-for-entity b" in context
    assert "triplet-for-entity c" in context
    assert "Context for :" not in context
