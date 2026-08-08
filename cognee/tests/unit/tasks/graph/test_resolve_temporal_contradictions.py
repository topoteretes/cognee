"""Unit tests for the temporal contradiction resolution task (issue #3631, Approach E).

Deterministic: no LLM, no database, no network — the graph engine is mocked.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.graph.resolve_temporal_contradictions import (
    _collect_touched_node_ids,
    resolve_temporal_contradictions,
)

rtc_module = importlib.import_module("cognee.tasks.graph.resolve_temporal_contradictions")


def _entity(node_id):
    return SimpleNamespace(id=node_id)


def _chunk(contains):
    """A raw DocumentChunk-like item: entities live directly on ``contains``."""
    return SimpleNamespace(contains=contains)


def _summary(contains):
    """A TextSummary-like item: the real pipeline shape, wrapping its chunk in
    ``made_from`` and carrying no ``contains`` of its own."""
    return SimpleNamespace(made_from=_chunk(contains))


def _props(updated_at, edge_object_id):
    return {"updated_at": updated_at, "edge_object_id": edge_object_id, "feedback_weight": 0.5}


# Acme's CEO was Alice (stored by an earlier ingestion); the new one says Bob.
NODES = [
    ("acme", {"name": "Acme"}),
    ("alice", {"name": "Alice"}),
    ("bob", {"name": "Bob"}),
    ("globex", {"name": "Globex"}),
]

EDGES = [
    ("acme", "alice", "ceo_of", _props("2020-01-01 00:00:00", "e-alice")),
    ("acme", "bob", "ceo_of", _props("2024-01-01 00:00:00", "e-bob")),
    # Many-valued relationship: both targets stay current.
    ("acme", "alice", "mentions", _props("2020-01-01 00:00:00", "m-alice")),
    ("acme", "bob", "mentions", _props("2024-01-01 00:00:00", "m-bob")),
    # A conflict this ingestion did not touch: Globex is only a neighbour.
    ("globex", "alice", "ceo_of", _props("2019-01-01 00:00:00", "g-alice")),
    ("globex", "bob", "ceo_of", _props("2023-01-01 00:00:00", "g-bob")),
]


def _mock_graph_engine(nodes=NODES, edges=EDGES):
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=(nodes, edges))
    engine.add_edges = AsyncMock()
    return engine


# --------------------------------------------------------------------------- helpers


def test_collect_touched_node_ids_handles_entities_and_tuples():
    chunk = _chunk([_entity("acme"), (MagicMock(), _entity("bob"))])
    assert _collect_touched_node_ids([chunk]) == {"acme", "bob"}


def test_collect_touched_node_ids_reads_through_made_from():
    # Regression: the real pipeline passes TextSummary objects (entities live on
    # ``made_from.contains``, not on the summary itself). Reading the top-level
    # object only would collect nothing and make the whole task a silent no-op.
    assert _collect_touched_node_ids([_summary([_entity("acme")])]) == {"acme"}


def test_collect_touched_node_ids_ignores_missing_contains():
    assert _collect_touched_node_ids([SimpleNamespace(contains=None)]) == set()


# --------------------------------------------------------------------------- task


@pytest.mark.asyncio
async def test_returns_input_when_no_data_points():
    assert await resolve_temporal_contradictions([], {"ceo_of"}) == []


@pytest.mark.asyncio
async def test_noop_without_functional_relationships():
    chunks = [_summary([_entity("acme"), _entity("bob")])]
    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock) as mock_engine:
        result = await resolve_temporal_contradictions(chunks)
    assert result is chunks
    mock_engine.assert_not_called()


@pytest.mark.asyncio
async def test_returns_early_when_nothing_touched():
    chunks = [SimpleNamespace(contains=[])]
    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock) as mock_engine:
        result = await resolve_temporal_contradictions(chunks, {"ceo_of"})
    assert result is chunks
    mock_engine.assert_not_called()


@pytest.mark.asyncio
async def test_supersedes_the_older_stored_assertion():
    # Real pipeline shape: TextSummary objects wrapping their chunk in made_from.
    chunks = [_summary([_entity("acme"), _entity("bob")])]
    engine = _mock_graph_engine()

    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine):
        result = await resolve_temporal_contradictions(chunks, {"ceo_of"})

    assert result is chunks  # pass-through, so the task can be appended anywhere

    # The touched entities seed the neighbourhood fetch.
    engine.get_neighborhood.assert_awaited_once()
    assert set(engine.get_neighborhood.await_args.args[0]) == {"acme", "bob"}

    # Only the outdated Acme fact is written back, tagged and pointing at the winner.
    engine.add_edges.assert_awaited_once()
    written = engine.add_edges.await_args.args[0]
    assert [(e[0], e[1], e[2]) for e in written] == [("acme", "alice", "ceo_of")]
    properties = written[0][3]
    assert properties["superseded"] is True
    assert properties["superseded_by"] == "e-bob"
    assert "ceo_of" in properties["supersession_reason"]
    # The stored properties survive the whole-blob rewrite.
    assert properties["updated_at"] == "2020-01-01 00:00:00"
    assert properties["feedback_weight"] == 0.5


@pytest.mark.asyncio
async def test_undeclared_relationships_are_never_collapsed():
    # Acme mentions both Alice and Bob. 'mentions' is not declared functional,
    # so the two targets are legitimate and neither is superseded.
    edges = [
        ("acme", "alice", "mentions", _props("2020-01-01 00:00:00", "m-alice")),
        ("acme", "bob", "mentions", _props("2024-01-01 00:00:00", "m-bob")),
    ]
    chunks = [_summary([_entity("acme")])]
    engine = _mock_graph_engine(edges=edges)

    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine):
        await resolve_temporal_contradictions(chunks, {"ceo_of"})

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_subjects_outside_this_ingestion_are_left_alone():
    # Globex holds the same conflict but was not touched: it is only a neighbour
    # of Alice/Bob, so resolving it is not this run's business.
    chunks = [_summary([_entity("bob")])]
    engine = _mock_graph_engine()

    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine):
        await resolve_temporal_contradictions(chunks, {"ceo_of"})

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_no_conflict_writes_nothing():
    edges = [("acme", "bob", "ceo_of", _props("2024-01-01 00:00:00", "e-bob"))]
    chunks = [_summary([_entity("acme")])]
    engine = _mock_graph_engine(edges=edges)

    with patch.object(rtc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine):
        await resolve_temporal_contradictions(chunks, {"ceo_of"})

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_errors_are_swallowed_and_input_returned():
    chunks = [_summary([_entity("acme")])]

    with patch.object(
        rtc_module,
        "get_graph_engine",
        new_callable=AsyncMock,
        side_effect=RuntimeError("graph down"),
    ):
        result = await resolve_temporal_contradictions(chunks, {"ceo_of"})

    assert result is chunks


# --------------------------------------------------------------------------- pipeline wiring


@pytest.mark.asyncio
async def test_task_is_appended_only_when_relationships_are_declared():
    from cognee.api.v1.cognify.cognify import get_default_tasks

    default_tasks = await get_default_tasks(chunk_size=1024)
    assert resolve_temporal_contradictions not in [task.executable for task in default_tasks]

    opted_in = await get_default_tasks(chunk_size=1024, functional_relationships={"ceo_of"})
    # Runs last, after the graph the facts are compared against has been written.
    assert opted_in[-1].executable is resolve_temporal_contradictions
    assert opted_in[-1].default_params["kwargs"]["functional_relationships"] == {"ceo_of"}
