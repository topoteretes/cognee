"""Unit tests for undo_forget operation and get_event_log."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.versioning.operations.undo_forget import UndoForgetResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    dataset_id,
    data_id=None,
    node_slugs=None,
    edge_slugs=None,
    datapoints=None,
    undone_at=None,
    expires_at=None,
    sequence_number=1,
):
    ev = MagicMock()
    ev.id = uuid4()
    ev.operation = "FORGET"
    ev.dataset_id = dataset_id
    ev.data_id = data_id
    ev.user_id = None
    ev.run_id = None
    ev.undone_at = undone_at
    ev.sequence_number = sequence_number
    # Default: still within retention window
    ev.expires_at = expires_at if expires_at is not None else (
        datetime.now(timezone.utc) + timedelta(days=30)
    )
    ev.payload = json.dumps({
        "node_slugs": [str(s) for s in (node_slugs or [])],
        "edge_slugs": edge_slugs or [],
        "datapoints": datapoints or [],
    })
    ev.created_at = datetime.now(timezone.utc)
    return ev


def _make_add_event(dataset_id, data_id=None, node_slugs=None, datapoints=None):
    ev = MagicMock()
    ev.id = uuid4()
    ev.operation = "ADD"
    ev.dataset_id = dataset_id
    ev.data_id = data_id
    ev.sequence_number = 1
    ev.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    ev.payload = json.dumps({
        "node_slugs": [str(s) for s in (node_slugs or [])],
        "edge_slugs": [],
        "datapoints": datapoints or [],
    })
    return ev


def _make_restore_seq_result(current_max=2):
    """Mock execute result for the RESTORE event sequence-number query."""
    r = MagicMock()
    r.scalar_one.return_value = current_max
    return r


def _make_session_for_forget(forget_event, add_events=None, rowcount=2):
    """Session: FORGET SELECT → UPDATE → ADD SELECT → RESTORE SEQ."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    forget_scalar = MagicMock()
    forget_scalar.scalar_one_or_none.return_value = forget_event

    update_result = MagicMock()
    update_result.rowcount = rowcount

    add_scalars = MagicMock()
    add_scalars.scalars.return_value.all.return_value = add_events or []

    restore_seq = _make_restore_seq_result()

    session.execute = AsyncMock(side_effect=[forget_scalar, update_result, add_scalars, restore_seq])
    return session


# ---------------------------------------------------------------------------
# UndoForgetResult
# ---------------------------------------------------------------------------


def test_undo_forget_result_to_dict():
    eid = uuid4()
    slugs = [str(uuid4()), str(uuid4())]
    r = UndoForgetResult(
        event_id=eid,
        node_slugs=slugs,
        edge_slugs=["rel_a"],
        ledger_rows_restored=3,
        datapoints_for_reingest=[{"id": str(uuid4()), "type": "Entity"}],
    )
    d = r.to_dict()
    assert d["event_id"] == str(eid)
    assert d["node_slugs"] == slugs
    assert d["ledger_rows_restored"] == 3
    assert d["datapoints_for_reingest_count"] == 1
    assert "note" in d


# ---------------------------------------------------------------------------
# undo_forget — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_forget_clears_ledger_and_marks_event():
    dataset_id = uuid4()
    data_id = uuid4()
    node_slugs = [uuid4(), uuid4()]
    event = _make_event(dataset_id, data_id=data_id, node_slugs=node_slugs)
    session = _make_session_for_forget(event, rowcount=2)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, data_id=data_id, session=session)

    assert isinstance(result, UndoForgetResult)
    assert result.ledger_rows_restored == 2
    assert result.node_slugs == [str(s) for s in node_slugs]
    assert event.undone_at is not None
    # session.add is called twice: once for FORGET event, once for RESTORE event
    session.add.assert_any_call(event)
    assert session.add.call_count == 2


@pytest.mark.asyncio
async def test_undo_forget_raises_when_no_event():
    dataset_id = uuid4()
    session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=scalar_result)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        with pytest.raises(ValueError, match="No unresolved FORGET event"):
            await undo_forget(dataset_id, session=session)


@pytest.mark.asyncio
async def test_undo_forget_raises_when_expired():
    """Raise ValueError when the event is past its retention window."""
    dataset_id = uuid4()
    expired_time = datetime.now(timezone.utc) - timedelta(days=1)
    event = _make_event(dataset_id, expires_at=expired_time)

    session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = event
    session.execute = AsyncMock(return_value=scalar_result)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        with pytest.raises(ValueError, match="has expired"):
            await undo_forget(dataset_id, session=session)


@pytest.mark.asyncio
async def test_undo_forget_with_empty_node_slugs():
    """An event with no node slugs skips UPDATE but still marks event undone."""
    dataset_id = uuid4()
    event = _make_event(dataset_id, node_slugs=[], edge_slugs=[])

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    forget_scalar = MagicMock()
    forget_scalar.scalar_one_or_none.return_value = event

    # With empty node_slugs: UPDATE and ADD-SELECT are skipped;
    # only FORGET SELECT + RESTORE SEQ execute are called (2 total).
    restore_seq = _make_restore_seq_result()
    session.execute = AsyncMock(side_effect=[forget_scalar, restore_seq])

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, session=session)

    # FORGET SELECT + RESTORE SEQ (UPDATE and ADD-SELECT skipped for empty slugs)
    assert session.execute.await_count == 2
    assert result.ledger_rows_restored == 0


@pytest.mark.asyncio
async def test_undo_forget_returns_datapoints_from_add_events():
    """undo_forget should surface DataPoint JSON from matching ADD events."""
    dataset_id = uuid4()
    node_id = uuid4()
    dp_json = json.dumps({"id": str(node_id), "type": "Entity", "name": "Paris"})

    forget_event = _make_event(dataset_id, node_slugs=[node_id])
    add_event = _make_add_event(
        dataset_id,
        node_slugs=[node_id],
        datapoints=[dp_json],
    )

    session = _make_session_for_forget(forget_event, add_events=[add_event], rowcount=1)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, session=session)

    assert len(result.datapoints_for_reingest) == 1
    assert result.datapoints_for_reingest[0]["name"] == "Paris"


# ---------------------------------------------------------------------------
# get_event_log
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_event_log_returns_dicts():
    dataset_id = uuid4()
    node_id = uuid4()
    event = _make_event(dataset_id, node_slugs=[node_id], edge_slugs=["rel_x"])
    event.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    event.undone_at = None
    event.data_id = None
    event.user_id = None
    event.run_id = None

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = [event]
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_mock

    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)

    with patch(
        "cognee.modules.versioning.operations.get_events.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.get_events import get_event_log

        events = await get_event_log(dataset_id, session=session)

    assert len(events) == 1
    ev = events[0]
    assert ev["operation"] == "FORGET"
    assert ev["dataset_id"] == str(dataset_id)
    assert "rel_x" in ev["edge_slugs"]


@pytest.mark.asyncio
async def test_get_event_log_empty_returns_empty_list():
    dataset_id = uuid4()

    scalars_mock = MagicMock()
    scalars_mock.all.return_value = []
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_mock

    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)

    with patch(
        "cognee.modules.versioning.operations.get_events.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.get_events import get_event_log

        events = await get_event_log(dataset_id, session=session)

    assert events == []


# ---------------------------------------------------------------------------
# RESTORE event logging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_undo_forget_appends_restore_event():
    """undo_forget must log a RESTORE VersionEvent with operation='RESTORE'."""
    dataset_id = uuid4()
    node_slugs = [uuid4(), uuid4()]
    event = _make_event(dataset_id, node_slugs=node_slugs)

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    forget_scalar = MagicMock()
    forget_scalar.scalar_one_or_none.return_value = event

    update_result = MagicMock()
    update_result.rowcount = 2

    add_scalars = MagicMock()
    add_scalars.scalars.return_value.all.return_value = []

    restore_seq = _make_restore_seq_result(current_max=2)
    session.execute = AsyncMock(
        side_effect=[forget_scalar, update_result, add_scalars, restore_seq]
    )

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        await undo_forget(dataset_id, session=session)

    # Two session.add calls: FORGET event (undone_at update) + RESTORE event
    assert session.add.call_count == 2
    added_objects = [call[0][0] for call in session.add.call_args_list]
    restore_events = [o for o in added_objects if getattr(o, "operation", None) == "RESTORE"]
    assert len(restore_events) == 1
    assert restore_events[0].sequence_number == 3  # MAX(2) + 1
    assert restore_events[0].dataset_id == dataset_id


# ---------------------------------------------------------------------------
# forget → undo preserves exact same UUIDs (issue requirement)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forget_undo_same_uuids():
    """result.node_slugs must be the exact UUID strings from the FORGET event."""
    dataset_id = uuid4()
    original_uuids = [uuid4(), uuid4(), uuid4()]
    event = _make_event(dataset_id, node_slugs=original_uuids)
    session = _make_session_for_forget(event, rowcount=3)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, session=session)

    assert result.node_slugs == [str(u) for u in original_uuids]
    assert len(result.node_slugs) == 3
