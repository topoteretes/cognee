"""Guards entity/result alignment in TripletSearchContextProvider.

``_get_search_tasks`` skips entities that have no searchable text, so the list
of search results is shorter than the original ``entities`` list whenever any
entity is text-less. ``get_context`` used to zip the *full* ``entities`` list
against those results, which mislabels every entity after the skipped one (and
silently drops the trailing entities via ``zip``). The provider now returns the
surviving entities alongside their tasks and zips those instead.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.retrieval.context_providers import TripletSearchContextProvider as mod
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


@pytest.mark.asyncio
async def test_context_pairs_each_entity_with_its_own_result(monkeypatch):
    # A text-less entity sits between two entities that do have text. On the
    # buggy code this shifts results so "Gamma" is dropped and pairs the empty
    # entity with Gamma's result.
    entities = [
        SimpleNamespace(name="Alpha"),
        SimpleNamespace(name=""),  # no searchable text -> skipped by _get_search_tasks
        SimpleNamespace(name="Gamma"),
    ]

    async def fake_triplet_search(query, **kwargs):
        # Echo the query (which embeds the entity text) so we can verify which
        # entity each result belongs to.
        return f"RESULT::{query}"

    async def fake_get_memory_fragment(_properties):
        return None

    monkeypatch.setattr(mod, "brute_force_triplet_search", fake_triplet_search)
    monkeypatch.setattr(mod, "get_memory_fragment", fake_get_memory_fragment)
    monkeypatch.setattr(mod, "format_triplets", lambda triplets: triplets)

    provider = TripletSearchContextProvider()
    context = await provider.get_context(entities, query="Q")

    # Each entity must be paired with the result derived from its OWN text.
    assert "Context for Alpha:\nRESULT::Alpha Q" in context
    assert "Context for Gamma:\nRESULT::Gamma Q" in context
    # Gamma must not be dropped, and its result must not leak onto another entity.
    assert "Context for Gamma:\nRESULT::Alpha Q" not in context
