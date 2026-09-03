"""extract_graph_and_summarize: the temporal lane is layered on the standard stages (SDK-80)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.engine.models import Entity, EntityType, Event
from cognee.tasks.graph import extract_graph_and_summarize as module

MODULE = "cognee.tasks.graph.extract_graph_and_summarize"


def _chunk(text):
    return SimpleNamespace(text=text, contains=None)


@pytest.mark.asyncio
async def test_default_runs_only_graph_and_summary_lanes():
    chunks = [_chunk("a"), _chunk("b")]
    summaries = ["summary-a", "summary-b"]

    with (
        patch(f"{MODULE}.extract_graph_from_data", AsyncMock(return_value=chunks)) as graph,
        patch(f"{MODULE}.summarize_text", AsyncMock(return_value=summaries)) as summarize,
        patch(f"{MODULE}.extract_events_for_chunks", AsyncMock()) as events,
    ):
        result = await module.extract_graph_and_summarize(chunks, graph_model=object)

    assert result == summaries
    graph.assert_awaited_once()
    summarize.assert_awaited_once()
    events.assert_not_awaited()
    assert all(chunk.contains is None for chunk in chunks)


@pytest.mark.asyncio
async def test_extract_events_attaches_after_graph_extraction_and_links_entities():
    """Graph extraction assigns chunk.contains outright; events must land after it and be
    linked to the chunk's own entities."""
    chunks = [_chunk("Einstein published in 1905."), _chunk("Nothing dated here.")]
    einstein = Entity(
        name="einstein",
        description="physicist",
        is_a=EntityType(name="person", type="person", description="p"),
    )
    event = Event(name="Einstein publishes", description="Einstein published in 1905.")

    async def fake_graph(data_chunks, **kwargs):
        data_chunks[0].contains = [einstein]  # overwrite, as the real task does
        data_chunks[1].contains = []
        return data_chunks

    with (
        patch(f"{MODULE}.extract_graph_from_data", fake_graph),
        patch(f"{MODULE}.summarize_text", AsyncMock(return_value=["s1", "s2"])),
        patch(f"{MODULE}.extract_events_for_chunks", AsyncMock(return_value=[[event], []])),
    ):
        result = await module.extract_graph_and_summarize(
            chunks, graph_model=object, extract_events=True
        )

    assert result == ["s1", "s2"]
    assert chunks[0].contains == [einstein, event]
    assert chunks[1].contains == []
    ((edge, targets),) = event.attributes
    assert edge.relationship_type == "involves"
    assert targets == [einstein]


@pytest.mark.asyncio
async def test_extract_events_flag_is_not_forwarded_to_graph_extraction():
    """The flag is a named parameter: it must not leak into the LLM kwargs."""
    chunks = [_chunk("a")]
    graph = AsyncMock(return_value=chunks)

    with (
        patch(f"{MODULE}.extract_graph_from_data", graph),
        patch(f"{MODULE}.summarize_text", AsyncMock(return_value=["s"])),
        patch(f"{MODULE}.extract_events_for_chunks", AsyncMock(return_value=[[]])),
    ):
        await module.extract_graph_and_summarize(
            chunks, graph_model=object, extract_events=True, custom_prompt="p"
        )

    assert "extract_events" not in graph.await_args.kwargs
    assert graph.await_args.kwargs["custom_prompt"] == "p"
