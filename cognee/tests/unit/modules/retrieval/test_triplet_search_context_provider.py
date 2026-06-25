"""Unit tests for TripletSearchContextProvider entity/result alignment."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


@pytest.mark.asyncio
async def test_get_context_aligns_results_with_text_bearing_entities():
    """Each search result must map to the entity that produced it.

    When an entity has no text, it is skipped during search task creation.
    Results must still be paired with the correct entities — not shifted via
    zip() against the full unfiltered list.
    """
    entity_a = SimpleNamespace(name="", description="", text="")
    entity_b = SimpleNamespace(name="EntityB", description="", text="")
    entity_c = SimpleNamespace(name="EntityC", description="", text="")
    entities = [entity_a, entity_b, entity_c]

    search_call_queries: list[str] = []
    format_calls: list[tuple[str, str]] = []

    async def mock_search(**kwargs):
        query = kwargs.get("query", "")
        search_call_queries.append(query)
        if "EntityB" in query:
            return "result-for-B"
        if "EntityC" in query:
            return "result-for-C"
        return "unexpected-result"

    async def mock_format_triplets(triplets, entity_name):
        format_calls.append((entity_name, triplets))
        return f"Context for {entity_name}:\n{triplets}\n---\n"

    provider = TripletSearchContextProvider()
    provider._format_triplets = mock_format_triplets

    with (
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.get_memory_fragment",
            new=AsyncMock(return_value=SimpleNamespace()),
        ),
        patch(
            "cognee.modules.retrieval.context_providers.TripletSearchContextProvider.brute_force_triplet_search",
            side_effect=mock_search,
        ),
    ):
        context = await provider.get_context(entities, query="test query")

    # Search runs only for text-bearing entities (B and C), in order.
    assert search_call_queries == ["EntityB test query", "EntityC test query"]

    # Correct behavior: B's result maps to EntityB, C's result maps to EntityC.
    assert format_calls == [
        ("EntityB", "result-for-B"),
        ("EntityC", "result-for-C"),
    ]

    assert "Context for EntityB:\nresult-for-B" in context
    assert "Context for EntityC:\nresult-for-C" in context
