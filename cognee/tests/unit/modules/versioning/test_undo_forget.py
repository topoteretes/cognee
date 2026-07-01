"""Unit tests for undo_forget operation and get_event_log."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.modules.versioning.operations.undo_forget import UndoForgetResult


# ---------------------------------------------------------------------------
# Helpers — use plain MagicMock, not SQLAlchemy ORM instantiation
# ---------------------------------------------------------------------------


def _make_event(dataset_id, data_id=None, node_slugs=None, edge_slugs=None, undone_at=None):
    ev = MagicMock()
    ev.id = uuid4()
    ev.operation = "FORGET"
    ev.dataset_id = dataset_id
    ev.data_id = data_id
    ev.user_id = None
    ev.run_id = None
    ev.undone_at = undone_at
    ev.payload = json.dumps(
        {
            "node_slugs": [str(s) for s in (node_slugs or [])],
            "edge_slugs": edge_slugs or [],
        }
    )
    ev.created_at = datetime.now(timezone.utc)
    return ev


def _make_session_with_event(event, rowcount=2):
    """Return a session mock whose first execute returns the event, second returns rowcount."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = event

    execute_result_update = MagicMock()
    execute_result_update.rowcount = rowcount

    session.execute = AsyncMock(side_effect=[scalar_result, execute_result_update])
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
    )
    d = r.to_dict()
    assert d["event_id"] == str(eid)
    assert d["node_slugs"] == slugs
    assert d["ledger_rows_restored"] == 3
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
    session = _make_session_with_event(event, rowcount=2)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, data_id=data_id, session=session)

    assert isinstance(result, UndoForgetResult)
    assert result.ledger_rows_restored == 2
    assert result.node_slugs == [str(s) for s in node_slugs]

    # Event should be marked undone
    assert event.undone_at is not None
    session.add.assert_called_with(event)
    session.flush.assert_awaited()


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
async def test_undo_forget_with_empty_node_slugs():
    """An event with no node slugs should not issue any UPDATE statements."""
    dataset_id = uuid4()
    event = _make_event(dataset_id, node_slugs=[], edge_slugs=[])

    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()

    scalar_result = MagicMock()
    scalar_result.scalar_one_or_none.return_value = event
    session.execute = AsyncMock(return_value=scalar_result)

    with patch(
        "cognee.modules.versioning.operations.undo_forget.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.undo_forget import undo_forget

        result = await undo_forget(dataset_id, session=session)

    # execute called once for SELECT, zero UPDATE calls (no slugs)
    assert session.execute.await_count == 1
    assert result.ledger_rows_restored == 0


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
    assert isinstance(ev["node_slugs"], list)
    assert isinstance(ev["edge_slugs"], list)
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
