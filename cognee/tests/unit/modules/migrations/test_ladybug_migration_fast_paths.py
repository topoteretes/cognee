"""Ladybug-only migration shortcuts (COG-6185).

Two phases dominated the fork re-key on a 100k-node graph: re-asserting every
at-risk edge (~102k upserts, ~22 min) and moving provenance key by key (~400k
point updates, ~11 min). Both are replaced by set-based statements the engine
can answer in one scan. These tests pin the behaviour that makes that safe:
the shortcuts must narrow the work, never skip work that is still needed, and
must fall back cleanly on backends that cannot answer them.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

import cognee.modules.migrations.versions.namespace_entity_type_node_ids as ns
import cognee.modules.migrations.versions.rekey_fork_document_ids as rk
from cognee.infrastructure.databases.provenance import make_source_ref_key


@pytest.mark.asyncio
async def test_only_dropped_edges_are_reasserted():
    """Edges the backend still holds must not be rewritten."""
    at_risk = [
        ("a", "b", "contains", {}),
        ("c", "d", "made_from", {}),
        ("e", "f", "is_part_of", {}),
    ]
    engine = AsyncMock()
    # The scan reports the first two survived; only the third vanished.
    engine.query.return_value = [["a", "b", "contains"], ["c", "d", "made_from"]]

    result = await ns._edges_needing_reassert(engine, at_risk)

    assert result == [("e", "f", "is_part_of", {})]


@pytest.mark.asyncio
async def test_nothing_reasserted_when_no_edge_was_dropped():
    at_risk = [("a", "b", "contains", {})]
    engine = AsyncMock()
    engine.query.return_value = [["a", "b", "contains"]]

    assert await ns._edges_needing_reassert(engine, at_risk) == []


@pytest.mark.asyncio
async def test_reassert_falls_back_to_everything_when_listing_unsupported():
    """A backend that cannot list edges keeps the old, exhaustive behaviour."""
    at_risk = [("a", "b", "contains", {}), ("c", "d", "made_from", {})]
    engine = AsyncMock()
    engine.query.side_effect = RuntimeError("Cypher not supported")

    assert await ns._edges_needing_reassert(engine, at_risk) == at_risk


@pytest.mark.asyncio
async def test_fast_provenance_move_rewrites_both_key_columns():
    """One statement per artifact kind, rewriting keys and run refs together."""
    engine = AsyncMock()

    moved = await rk._fast_move_provenance_ladybug(engine, "OLD_KEY", "NEW_KEY")

    assert moved is True
    assert engine.query.await_count == 2
    node_query, node_params = engine.query.await_args_list[0].args
    edge_query, _ = engine.query.await_args_list[1].args
    assert "MATCH (n:Node)" in node_query and "MATCH ()-[r:EDGE]->()" in edge_query
    for query in (node_query, edge_query):
        assert "source_ref_keys = replace(" in query
        assert "source_run_refs = replace(" in query
        # Artifacts already carrying the new key are excluded: a blind replace
        # would leave the key twice.
        assert "NOT" in query and "CONTAINS $new_key" in query
    assert node_params == {"old_key": "OLD_KEY", "new_key": "NEW_KEY"}


@pytest.mark.asyncio
async def test_fast_provenance_move_reports_failure_for_fallback():
    engine = AsyncMock()
    engine.query.side_effect = RuntimeError("function replace does not exist")

    assert await rk._fast_move_provenance_ladybug(engine, "OLD", "NEW") is False


@pytest.mark.asyncio
async def test_rekey_skips_generic_sweep_once_fast_move_clears_the_key():
    """With nothing left carrying the old key, the per-artifact path is skipped."""
    old_id, new_id, dataset_id = str(uuid4()), str(uuid4()), str(uuid4())
    engine = AsyncMock()
    engine.find_nodes_by_source_ref.return_value = []
    engine.find_edges_by_source_ref.return_value = []

    with (
        patch.object(rk, "stores_provenance_in_graph", new=AsyncMock(return_value=True)),
        patch.object(rk, "_is_ladybug_graph", return_value=True),
        patch.object(rk, "_fast_move_provenance_ladybug", new=AsyncMock(return_value=True)),
    ):
        await rk._rekey_graph_provenance(engine, [(old_id, new_id, dataset_id)])

    engine.get_node_delete_data.assert_not_awaited()
    engine.attach_node_source_refs.assert_not_awaited()
    engine.remove_node_source_refs.assert_not_awaited()


@pytest.mark.asyncio
async def test_rekey_still_sweeps_artifacts_the_fast_move_left_behind():
    """Artifacts holding both keys (interrupted run) go through the generic path."""
    old_id, new_id, dataset_id = str(uuid4()), str(uuid4()), str(uuid4())
    old_key = make_source_ref_key(dataset_id, __import__("uuid").UUID(old_id))
    run = "00000000-0000-0000-0000-000000000001"
    snapshot = SimpleNamespace(source_run_refs=[f"source_run_ref:v1:{run}:{old_key}"])

    engine = AsyncMock()
    engine.find_nodes_by_source_ref.return_value = ["leftover-node"]
    engine.find_edges_by_source_ref.return_value = []
    engine.get_node_delete_data.return_value = {"leftover-node": snapshot}

    with (
        patch.object(rk, "stores_provenance_in_graph", new=AsyncMock(return_value=True)),
        patch.object(rk, "_is_ladybug_graph", return_value=True),
        patch.object(rk, "_fast_move_provenance_ladybug", new=AsyncMock(return_value=True)),
    ):
        await rk._rekey_graph_provenance(engine, [(old_id, new_id, dataset_id)])

    engine.attach_node_source_refs.assert_awaited_once()
    engine.remove_node_source_refs.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_ladybug_backends_keep_the_generic_path():
    old_id, new_id, dataset_id = str(uuid4()), str(uuid4()), str(uuid4())
    engine = AsyncMock()
    engine.find_nodes_by_source_ref.return_value = []
    engine.find_edges_by_source_ref.return_value = []
    fast = AsyncMock(return_value=True)

    with (
        patch.object(rk, "stores_provenance_in_graph", new=AsyncMock(return_value=True)),
        patch.object(rk, "_is_ladybug_graph", return_value=False),
        patch.object(rk, "_fast_move_provenance_ladybug", new=fast),
    ):
        await rk._rekey_graph_provenance(engine, [(old_id, new_id, dataset_id)])

    fast.assert_not_awaited()
