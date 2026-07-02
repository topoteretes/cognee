import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee.tasks.graph.detect_contradictions import (
    Contradiction,
    ContradictionList,
    _build_candidate_facts,
    _collect_touched_node_ids,
    _contradiction_endpoints,
    _node_names,
    detect_contradictions,
)

dc_module = importlib.import_module("cognee.tasks.graph.detect_contradictions")


def _entity(node_id):
    return SimpleNamespace(id=node_id)


def _chunk(contains):
    return SimpleNamespace(contains=contains)


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
    engine.get_graph_data = AsyncMock(return_value=(NODES, EDGES))
    engine.add_edges = AsyncMock()
    return engine


# --------------------------------------------------------------------------- helpers


def test_collect_touched_node_ids_handles_entities_and_tuples():
    chunk = _chunk([_entity("alice"), (MagicMock(), _entity("1990"))])
    assert _collect_touched_node_ids([chunk]) == {"alice", "1990"}


def test_collect_touched_node_ids_ignores_missing_contains():
    assert _collect_touched_node_ids([SimpleNamespace(contains=None)]) == set()


def test_node_names_skips_unnamed_nodes():
    names = _node_names(NODES)
    assert names["alice"] == "Alice"
    assert "chunk-1" not in names


def test_build_candidate_facts_filters_to_touched_named_edges():
    names = _node_names(NODES)
    lines, edge_by_id = _build_candidate_facts(EDGES, names, {"alice", "1990"}, limit=500)

    # Both Alice facts are kept; the Bob fact (untouched) and the structural edge are dropped.
    assert len(lines) == 2
    assert edge_by_id["F0"] == ("alice", "1985")
    assert edge_by_id["F1"] == ("alice", "1990")
    assert "Alice born in 1985" in lines[0]


def test_build_candidate_facts_respects_limit():
    names = _node_names(NODES)
    lines, _ = _build_candidate_facts(EDGES, names, {"alice"}, limit=1)
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
    chunks = [_chunk([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    llm_result = ContradictionList(
        contradictions=[
            Contradiction(
                first_fact_id="F0",
                second_fact_id="F1",
                first_fact="Alice born in 1985",
                second_fact="Alice born in 1990",
                reason="A person has a single birth year.",
                confidence=0.95,
            )
        ]
    )

    with (
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
    engine.add_edges.assert_awaited_once()
    added_edges = engine.add_edges.await_args.args[0]
    assert len(added_edges) == 1
    source, target, relationship, properties = added_edges[0]
    # Shared subject (Alice) -> the disagreement is between the objects.
    assert {source, target} == {"1985", "1990"}
    assert relationship == "contradicts"
    assert properties["reason"] == "A person has a single birth year."
    assert properties["first_fact"] == "Alice born in 1985"


@pytest.mark.asyncio
async def test_low_confidence_contradiction_is_not_flagged():
    chunks = [_chunk([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    llm_result = ContradictionList(
        contradictions=[
            Contradiction(
                first_fact_id="F0",
                second_fact_id="F1",
                first_fact="Alice born in 1985",
                second_fact="Alice born in 1990",
                reason="Maybe.",
                confidence=0.2,
            )
        ]
    )

    with (
        patch.object(dc_module, "get_graph_engine", new_callable=AsyncMock, return_value=engine),
        patch.object(
            dc_module.LLMGateway,
            "acreate_structured_output",
            new_callable=AsyncMock,
            return_value=llm_result,
        ),
    ):
        await detect_contradictions(chunks, confidence_threshold=0.5)

    engine.add_edges.assert_not_called()


@pytest.mark.asyncio
async def test_no_contradictions_does_not_write_edges():
    chunks = [_chunk([_entity("alice"), _entity("1990")])]
    engine = _mock_graph_engine()

    with (
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
    chunks = [_chunk([_entity("alice"), _entity("1990")])]

    with patch.object(
        dc_module,
        "get_graph_engine",
        new_callable=AsyncMock,
        side_effect=RuntimeError("graph down"),
    ):
        result = await detect_contradictions(chunks)

    assert result == chunks
