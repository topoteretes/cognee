"""Tests for get_nodes_at_time (time-travel via checkpoint + event replay).

Covers the issue-required scenarios:
  - snapshot → mutate → time-travel (exact node set)
  - compaction-safe time-travel (checkpoint as base, no early events needed)
  - FORGET undone before target_time → node stays alive
  - empty dataset → empty result
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc(*args) -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime(*args, tzinfo=timezone.utc)


def _make_checkpoint(dataset_id, node_slugs, created_at):
    cp = MagicMock()
    cp.id = uuid4()
    cp.dataset_id = dataset_id
    cp.node_slugs = json.dumps([str(s) for s in node_slugs])
    cp.edge_slugs = json.dumps([])
    cp.created_at = created_at
    return cp


def _make_event(dataset_id, operation, node_slugs, created_at, seq, undone_at=None):
    ev = MagicMock()
    ev.id = uuid4()
    ev.dataset_id = dataset_id
    ev.operation = operation
    ev.sequence_number = seq
    ev.created_at = created_at
    ev.undone_at = undone_at
    ev.payload = json.dumps({
        "node_slugs": [str(s) for s in node_slugs],
        "edge_slugs": [],
        "datapoints": [],
    })
    return ev


def _session_with_responses(cp, events):
    """Session that returns *cp* for the checkpoint query and *events* for replay."""
    session = AsyncMock()

    cp_result = MagicMock()
    cp_result.scalar_one_or_none.return_value = cp

    events_scalars = MagicMock()
    events_scalars.scalars.return_value.all.return_value = events

    session.execute = AsyncMock(side_effect=[cp_result, events_scalars])
    return session


# ---------------------------------------------------------------------------
# snapshot → mutate → time-travel (exact node set)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snapshot_mutate_time_travel_exact_node_set():
    """Time-travel to before a FORGET returns the original full node set."""
    dataset_id = uuid4()
    node_a, node_b = uuid4(), uuid4()

    t_add = _utc(2026, 1, 1, 9, 0)
    t_forget = _utc(2026, 1, 1, 11, 0)
    t_target = _utc(2026, 1, 1, 10, 0)  # between ADD and FORGET

    add_event = _make_event(dataset_id, "ADD", [node_a, node_b], t_add, seq=1)
    # FORGET happens after target — should NOT be applied
    forget_event = _make_event(dataset_id, "FORGET", [node_b], t_forget, seq=2)

    # No checkpoint before target
    session = _session_with_responses(cp=None, events=[add_event])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target, session=session)

    assert set(result["alive_node_slugs"]) == {str(node_a), str(node_b)}
    assert result["checkpoint_id"] is None


@pytest.mark.asyncio
async def test_time_travel_after_forget_excludes_forgotten_node():
    """Time-travel to after a FORGET returns the reduced set."""
    dataset_id = uuid4()
    node_a, node_b = uuid4(), uuid4()

    t_add = _utc(2026, 1, 1, 9, 0)
    t_forget = _utc(2026, 1, 1, 11, 0)
    t_target = _utc(2026, 1, 1, 12, 0)  # after both events

    add_event = _make_event(dataset_id, "ADD", [node_a, node_b], t_add, seq=1)
    forget_event = _make_event(dataset_id, "FORGET", [node_b], t_forget, seq=2)

    session = _session_with_responses(cp=None, events=[add_event, forget_event])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target, session=session)

    assert str(node_a) in result["alive_node_slugs"]
    assert str(node_b) not in result["alive_node_slugs"]


# ---------------------------------------------------------------------------
# compaction-safe time-travel (checkpoint as base)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_safe_time_travel_uses_checkpoint():
    """Works correctly even when events before the checkpoint are gone (compacted)."""
    dataset_id = uuid4()
    node_a, node_b, node_c = uuid4(), uuid4(), uuid4()

    t_cp = _utc(2026, 1, 1, 8, 0)
    t_target = _utc(2026, 1, 1, 14, 0)

    # Checkpoint holds node_a + node_b; early ADD events would be compacted
    cp = _make_checkpoint(dataset_id, [node_a, node_b], t_cp)
    # After checkpoint: node_c added, node_b forgotten
    add_after = _make_event(dataset_id, "ADD", [node_c], _utc(2026, 1, 1, 10, 0), seq=5)
    forget_after = _make_event(dataset_id, "FORGET", [node_b], _utc(2026, 1, 1, 12, 0), seq=6)

    session = _session_with_responses(cp=cp, events=[add_after, forget_after])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target, session=session)

    assert set(result["alive_node_slugs"]) == {str(node_a), str(node_c)}
    assert result["checkpoint_id"] == str(cp.id)
    assert result["checkpoint_time"] is not None


# ---------------------------------------------------------------------------
# FORGET undone before target_time → node stays alive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_time_travel_undone_forget_does_not_remove_node():
    """A FORGET that was undone before target_time must NOT subtract the node."""
    dataset_id = uuid4()
    node_a = uuid4()

    t_add = _utc(2026, 1, 1, 9, 0)
    t_forget = _utc(2026, 1, 1, 10, 0)
    t_undo = _utc(2026, 1, 1, 11, 0)   # undone before target
    t_target = _utc(2026, 1, 1, 12, 0)

    add_event = _make_event(dataset_id, "ADD", [node_a], t_add, seq=1)
    forget_event = _make_event(
        dataset_id, "FORGET", [node_a], t_forget, seq=2, undone_at=t_undo
    )

    session = _session_with_responses(cp=None, events=[add_event, forget_event])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target, session=session)

    assert str(node_a) in result["alive_node_slugs"]


# ---------------------------------------------------------------------------
# run-scoped rollback: undo_forget by run_id narrows scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_time_travel_run_scoped_rollback():
    """Nodes added in run_2 survive even when run_1 is rolled back (time-travel)."""
    dataset_id = uuid4()
    node_run1_a, node_run1_b = uuid4(), uuid4()
    node_run2 = uuid4()

    t_run1_add = _utc(2026, 1, 1, 9, 0)
    t_run1_forget = _utc(2026, 1, 1, 10, 0)
    t_run2_add = _utc(2026, 1, 1, 9, 30)
    t_target_before_run1_forget = _utc(2026, 1, 1, 9, 45)

    run1_add = _make_event(dataset_id, "ADD", [node_run1_a, node_run1_b], t_run1_add, seq=1)
    run2_add = _make_event(dataset_id, "ADD", [node_run2], t_run2_add, seq=2)

    # Time-travel to before run1's FORGET — all nodes alive
    session = _session_with_responses(cp=None, events=[run1_add, run2_add])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target_before_run1_forget, session=session)

    alive = set(result["alive_node_slugs"])
    assert str(node_run1_a) in alive
    assert str(node_run1_b) in alive
    assert str(node_run2) in alive


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_time_travel_empty_dataset_returns_empty():
    dataset_id = uuid4()
    t_target = _utc(2026, 6, 1, 12, 0)

    session = _session_with_responses(cp=None, events=[])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, t_target, session=session)

    assert result["alive_node_slugs"] == []
    assert result["checkpoint_id"] is None
    assert result["dataset_id"] == str(dataset_id)


@pytest.mark.asyncio
async def test_time_travel_naive_target_time_treated_as_utc():
    """Naive target_time (no tzinfo) should be treated as UTC without error."""
    dataset_id = uuid4()
    naive_target = datetime(2026, 6, 1, 12, 0)  # no timezone

    session = _session_with_responses(cp=None, events=[])

    with patch(
        "cognee.modules.versioning.operations.time_travel.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.time_travel import get_nodes_at_time

        result = await get_nodes_at_time(dataset_id, naive_target, session=session)

    assert "+00:00" in result["target_time"]  # normalised to UTC
