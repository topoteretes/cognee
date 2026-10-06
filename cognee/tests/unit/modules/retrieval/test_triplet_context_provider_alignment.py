"""Guards entity/result alignment in TripletSearchContextProvider.

Entities without searchable text are skipped, so there are fewer search results
than entities whenever one is text-less. ``get_context`` used to zip the *full*
``entities`` list against those results, which labelled every entity after the
skipped one with its neighbour's result and silently dropped the last one. The
provider now derives entities, searches and labels from one list of searchable
entities.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.retrieval.context_providers import TripletSearchContextProvider as mod
from cognee.modules.retrieval.context_providers.TripletSearchContextProvider import (
    TripletSearchContextProvider,
)


def _block(label: str, result: str) -> str:
    return f"Context for {label}:\n{result}\n---\n"


@pytest.fixture
def fake_search(monkeypatch):
    """Echoes each query back as its result and records projection loads."""
    calls = SimpleNamespace(memory_fragment=0, failing_queries=set())

    async def fake_triplet_search(query, **kwargs):
        if query in calls.failing_queries:
            raise RuntimeError(f"search failed for {query}")
        return f"RESULT::{query}"

    async def fake_get_memory_fragment(_properties):
        calls.memory_fragment += 1

    monkeypatch.setattr(mod, "brute_force_triplet_search", fake_triplet_search)
    monkeypatch.setattr(mod, "get_memory_fragment", fake_get_memory_fragment)
    monkeypatch.setattr(mod, "format_triplets", lambda triplets: triplets)
    return calls


@pytest.mark.asyncio
async def test_context_pairs_each_entity_with_its_own_result(fake_search):
    # A text-less entity between two searchable ones. The old code labelled it
    # "namespace(name='')" with Gamma's result and dropped Gamma entirely.
    entities = [
        SimpleNamespace(name="Alpha"),
        SimpleNamespace(name=""),
        SimpleNamespace(name="Gamma"),
    ]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == "\n".join(
        [_block("Alpha", "RESULT::Alpha Q"), _block("Gamma", "RESULT::Gamma Q")]
    )
    assert "namespace(" not in context


@pytest.mark.asyncio
async def test_whitespace_only_entities_are_skipped(fake_search):
    entities = [SimpleNamespace(name="   ", description="\n\t"), SimpleNamespace(name=" Beta ")]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == _block("Beta", "RESULT::Beta Q")


@pytest.mark.asyncio
async def test_label_is_the_first_text_field_of_the_search_text(fake_search):
    # No name: the search text starts with the description, so the block is
    # labelled with it rather than falling back to the object's repr.
    entities = [SimpleNamespace(name=None, description="Delta desc", text="Delta body")]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == _block("Delta desc", "RESULT::Delta desc Delta body Q")


@pytest.mark.asyncio
async def test_all_text_less_entities_skip_the_graph_projection(fake_search):
    entities = [SimpleNamespace(name=""), SimpleNamespace(description="  ")]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == "No valid entities found for context search."
    assert fake_search.memory_fragment == 0


@pytest.mark.asyncio
async def test_one_failed_search_keeps_the_other_entities_context(fake_search):
    fake_search.failing_queries = {"Alpha Q"}
    entities = [SimpleNamespace(name="Alpha"), SimpleNamespace(name="Gamma")]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    assert context == _block("Gamma", "RESULT::Gamma Q")


@pytest.mark.asyncio
async def test_every_search_failing_raises(fake_search):
    fake_search.failing_queries = {"Alpha Q", "Gamma Q"}
    entities = [SimpleNamespace(name="Alpha"), SimpleNamespace(name="Gamma")]

    with pytest.raises(RuntimeError, match="search failed"):
        await TripletSearchContextProvider().get_context(entities, query="Q")


@pytest.mark.asyncio
async def test_long_label_is_capped_but_search_text_is_not(fake_search):
    # A name-less entity with a long, multi-line description: the search uses
    # the full text, the block header is one line capped with an ellipsis.
    description = "first line\n" + " ".join(f"word{i}" for i in range(60))
    entities = [SimpleNamespace(name=None, description=description)]

    context = await TripletSearchContextProvider().get_context(entities, query="Q")

    label = " ".join(description.split())[: mod.MAX_LABEL_LENGTH - 1] + "…"
    assert context.startswith(f"Context for {label}:\n")
    assert f"RESULT::{description} Q" in context
