"""Regression tests for pairing searchable entities with their own results."""

import pytest

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.retrieval.context_providers import TripletSearchContextProvider as mod
from cognee.modules.retrieval.context_providers.SummarizedTripletSearchContextProvider import (
    SummarizedTripletSearchContextProvider,
)
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


def _block(label: str, result: str) -> str:
    return f"Context for {label}:\n{result}\n---\n"


@pytest.fixture
def fake_search(monkeypatch):
    """Echo queries so tests can identify which entity each result belongs to."""
    queries = []

    async def fake_triplet_search(query, **kwargs):
        queries.append(query)
        return f"RESULT::{query}"

    async def fake_get_memory_fragment(_properties):
        return None

    monkeypatch.setattr(mod, "brute_force_triplet_search", fake_triplet_search)
    monkeypatch.setattr(mod, "get_memory_fragment", fake_get_memory_fragment)
    monkeypatch.setattr(mod, "format_triplets", lambda triplets: triplets)
    return queries


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_position", [0, 1, 2])
async def test_context_pairs_each_entity_with_its_own_result(fake_search, empty_position):
    entities = [Entity(name="Alpha", description=""), Entity(name="Gamma", description="")]
    # The actual Entity schema accepts blank strings; no stand-in object is needed.
    entities.insert(empty_position, Entity(name="", description=""))

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == "\n".join(
        [_block("Alpha", "RESULT::Alpha Q"), _block("Gamma", "RESULT::Gamma Q")]
    )
    assert fake_search == ["Alpha Q", "Gamma Q"]
    assert len(entities) == 3


@pytest.mark.asyncio
async def test_generic_data_point_without_text_does_not_shift_results(fake_search):
    entities = [DataPoint(), Entity(name="Gamma", description="")]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == _block("Gamma", "RESULT::Gamma Q")
    assert fake_search == ["Gamma Q"]


@pytest.mark.asyncio
async def test_all_text_less_entities_keep_existing_response(fake_search):
    entities = [Entity(name="", description=""), DataPoint()]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == "No valid entities found for context search."
    assert fake_search == []


@pytest.mark.asyncio
async def test_description_fallback_keeps_full_label_and_search_text(fake_search):
    description = "first line\n" + " ".join(f"word{i}" for i in range(60))
    entities = [Entity(name="", description=""), Entity(name="", description=description)]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == _block(description, f"RESULT::{description} Q")
    assert fake_search == [f"{description} Q"]


@pytest.mark.asyncio
async def test_summarized_provider_inherits_alignment_fix(fake_search, monkeypatch):
    from cognee.modules.retrieval.context_providers import (
        SummarizedTripletSearchContextProvider as summary_mod,
    )

    async def fake_summary(text, _prompt):
        return f"SUMMARIZED::{text}"

    monkeypatch.setattr(summary_mod, "summarize_text", fake_summary)
    entities = [Entity(name="", description=""), Entity(name="Gamma", description="")]

    context = await SummarizedTripletSearchContextProvider().get_context(entities, query="Q")

    assert context == f"Summary for Gamma:\nSUMMARIZED::{_block('Gamma', 'RESULT::Gamma Q')}\n---\n"
    assert fake_search == ["Gamma Q"]
