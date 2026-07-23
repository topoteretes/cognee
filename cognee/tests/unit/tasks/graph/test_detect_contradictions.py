import importlib
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.modules.cognify.config import CognifyConfig
from cognee.tasks.graph.detect_contradictions import (
    _build_candidate_facts,
    _collect_touched_node_ids,
    _contradiction_endpoints,
    _node_names,
    detect_contradictions,
)
from cognee.tasks.graph.models import Contradiction, ContradictionList

dc_module = importlib.import_module("cognee.tasks.graph.detect_contradictions")


@contextmanager
def _patched_config(**overrides):
    """Pin the task's tuning so a local .env can never sway these assertions.

    The task reads its thresholds through get_cognify_config() rather than from
    call-site arguments, and constructor kwargs outrank env/dotenv in
    pydantic-settings — so every tunable is set explicitly, not just the overrides.
    """
    tuning = {"contradiction_confidence_threshold": 0.5, "contradiction_max_facts": 500}
    tuning.update(overrides)
    with patch.object(dc_module, "get_cognify_config", return_value=CognifyConfig(**tuning)):
        yield


def _entity(node_id):
    return SimpleNamespace(id=node_id)


def _chunk(contains):
    """A raw DocumentChunk-like item: entities live directly on ``contains``."""
    return SimpleNamespace(contains=contains)


def _summary(contains):
    """A TextSummary-like item: the real pipeline shape, wrapping its chunk in
    ``made_from`` and carrying no ``contains`` of its own."""
    return SimpleNamespace(made_from=_chunk(contains))


# Alice was born in 1985 (stored) but the new data claims 1990.
NODES = [
    ("alice", {"name": "Alice", "type": "Person"}),
    ("1985", {"name": "1985", "type": "Year"}),
    ("1990", {"name": "1990", "type": "Year"}),
    ("bob", {"name": "Bob", "type": "Person"}),
    ("chunk-1", {"type": "DocumentChunk"}),  # unnamed structural node
]

EDGES = [
    ("alice", "1985", "born_in", {}),
    ("alice", "1990", "born_in", {}),
    ("bob", "1985", "born_in", {}),  # untouched fact (neither endpoint touched)
    ("chunk-1", "alice", "contains", {}),  # structural edge, must be ignored
]


def _mock_graph_engine():
    engine = MagicMock()
    engine.get_neighborhood = AsyncMock(return_value=(NODES, EDGES))
    engine.add_edges = AsyncMock()
    return engine


# --------------------------------------------------------------------------- helpers


def test_collect_touched_node_ids_handles_entities_and_tuples():
    chunk = _chunk([_entity("alice"), (MagicMock(), _entity("1990"))])
    assert _collect_touched_node_ids([chunk]) == {"alice", "1990"}


def test_collect_touched_node_ids_reads_through_made_from():
    # Regression: the real pipeline passes TextSummary objects (entities live on
    # ``made_from.contains``, not on the summary itself). Reading the top-level
    # object only would collect nothing and make the whole task a silent no-op.
    assert _collect_touched_node_ids([_summary([_entity("alice")])]) == {"alice"}


def test_collect_touched_node_ids_ignores_missing_contains():
    assert _collect_touched_node_ids([SimpleNamespace(contains=None)]) == set()


def test_node_names_skips_unnamed_nodes():
    names = _node_names(NODES)
    assert names["alice"] == "Alice"
    assert "chunk-1" not in names


def test_build_candidate_facts_filters_to_touched_named_edges():
    names = _node_names(NODES)
    lines, fact_text, fact_edge = _build_candidate_facts(EDGES, names, {"alice", "1990"}, limit=500)

    # Both Alice facts are kept; the Bob fact (untouched) and the structural edge are dropped.
    assert len(lines) == 2
    assert fact_edge["F0"] == ("alice", "1985")
    assert fact_edge["F1"] == ("alice", "1990")
    assert fact_text["F0"] == "Alice born in 1985"
    assert "Alice born in 1985" in lines[0]


def test_build_candidate_facts_respects_limit():
    names = _node_names(NODES)
    lines, _, _ = _build_candidate_facts(EDGES, names, {"alice"}, limit=1)
    assert len(lines) == 1


def test_contradiction_endpoints_prefers_differing_objects_when_subject_shared():
    assert _contradiction_endpoints(("alice", "1985"), ("alice", "1990")) == ("1985", "1990")


def test_contradiction_endpoints_uses_subjects_when_they_differ():
    assert _contradiction_endpoints(("alice", "x"), ("bob", "x")) == ("alice", "bob")


def test_contradiction_endpoints_none_for_identical_edge():
    assert _contradiction_endpoints(("alice", "1985"), ("alice", "1985")) is None


# --------------------------------------------------------------------------- task


@pytest.mark.asyncio
async def test_returns_input_when_no_data_chunks():
    assert await detect_contradictions([]) == []


@pytest.mark.asyncio
async def test_returns_early_when_nothing_touched():
    chunks = [SimpleNamespace(contains=[])]
    with patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock) as mock_engine:
        result = await detect_contradictions(chunks)
    assert result is chunks
    mock_engine.assert_not_called()


@pytest.mark.asyncio
async def test_flags_contradiction_as_graph_edge():
    # Real pipeline shape: TextSummary objects wrapping their chunk in made_from.
    chunks = [_summary([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    llm_result = ContradictionList(
        contradictions=[
            Contradiction(
                first_fact_id="F0",
                second_fact_id="F1",
                reason="A person has a single birth year.",
                confidence=0.95,
            )
        ]
    )

    with (
        _patched_config(),
        patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine),
        patch.object(
            dc_module.LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=llm_result,
        ),
    ):
        result = await detect_contradictions(chunks)

    assert result == chunks  # non-destructive pass-through
    engine.get_neighborhood.assert_awaited_once()
    # The touched entities (from made_from.contains) seed the neighbourhood fetch.
    seeds = set(engine.get_neighborhood.await_args.args[0])
    assert seeds == {"alice", "1990"}

    engine.add_edges.assert_awaited_once()
    added_edges = engine.add_edges.await_args.args[0]
    assert len(added_edges) == 1
    source, target, relationship, properties = added_edges[0]
    # Shared subject (Alice) -> the disagreement is between the objects.
    assert {source, target} == {"1985", "1990"}
    assert relationship == "contradicts"
    assert properties["reason"] == "A person has a single birth year."
    # Fact text is reconstructed locally from the graph, not echoed by the LLM.
    assert properties["first_fact"] == "Alice born in 1985"
    assert properties["second_fact"] == "Alice born in 1990"


@pytest.mark.asyncio
async def test_low_confidence_contradiction_is_not_flagged():
    chunks = [_summary([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    llm_result = ContradictionList(
        contradictions=[
            Contradiction(
                first_fact_id="F0",
                second_fact_id="F1",
                reason="Maybe.",
                confidence=0.2,
            )
        ]
    )

    with (
        _patched_config(),
        patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine),
        patch.object(
            dc_module.LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=llm_result,
        ),
    ):
        await detect_contradictions(chunks)

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_max_facts_config_caps_the_facts_sent_to_the_llm():
    # The cap is config-driven, not a call-site argument: capping at one fact leaves
    # too few to compare, so the task returns before ever reaching the LLM.
    chunks = [_summary([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    with (
        _patched_config(contradiction_max_facts=1),
        patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine),
        patch.object(
            dc_module.LLMGateway, "acreate_structured_output", new_callable=AsyncMock
        ) as mock_llm,
    ):
        await detect_contradictions(chunks)

    mock_llm.assert_not_called()
    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_no_contradictions_does_not_write_edges():
    chunks = [_summary([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    with (
        _patched_config(),
        patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine),
        patch.object(
            dc_module.LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=ContradictionList(contradictions=[]),
        ),
    ):
        await detect_contradictions(chunks)

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_errors_are_swallowed_and_input_returned():
    chunks = [_summary([_entity("alice"), _entity("1990")])]

    with patch.object(
        dc_module,
        "get_graph_engine",
        new_callable=AsyncMock,
        side_effect=RuntimeError("graph down"),
    ):
        result = await detect_contradictions(chunks)

    assert result == chunks
