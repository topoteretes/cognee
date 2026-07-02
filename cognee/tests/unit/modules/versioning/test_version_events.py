"""Unit tests for VersionEvent model and log_version_event operation."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_mock():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    # execute returns a result whose scalar_one() gives next sequence number
    scalar_result = MagicMock()
    scalar_result.scalar_one.return_value = 0  # max_seq = 0 → next = 1
    session.execute = AsyncMock(return_value=scalar_result)
    return session


# ---------------------------------------------------------------------------
# VersionEvent model — metadata only (no ORM instantiation)
# ---------------------------------------------------------------------------


def test_version_event_tablename():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    assert VersionEvent.__tablename__ == "version_events"


def test_version_event_has_required_columns():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    cols = {c.name for c in VersionEvent.__table__.columns}
    for required in (
        "id", "operation", "dataset_id", "data_id",
        "created_at", "undone_at", "payload",
        "sequence_number", "expires_at",
    ):
        assert required in cols, f"Missing column: {required}"


def test_version_event_has_indexes():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    index_names = {idx.name for idx in VersionEvent.__table__.indexes}
    assert "idx_version_events_dataset_id" in index_names
    assert "idx_version_events_operation" in index_names
    assert "idx_version_events_sequence" in index_names


def test_default_retention_days():
    from cognee.modules.versioning.models.VersionEvent import DEFAULT_RETENTION_DAYS

    assert DEFAULT_RETENTION_DAYS == 30


# ---------------------------------------------------------------------------
# log_version_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_version_event_adds_and_returns_event():
    dataset_id = uuid4()
    data_id = uuid4()
    node_slugs = [str(uuid4()), str(uuid4())]
    edge_slugs = ["relates_to", "links_to"]

    session_mock = _make_session_mock()

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event(
            "FORGET",
            dataset_id,
            data_id=data_id,
            node_slugs=node_slugs,
            edge_slugs=edge_slugs,
            session=session_mock,
        )

    session_mock.add.assert_called_once()
    session_mock.flush.assert_awaited_once()
    added_obj = session_mock.add.call_args[0][0]
    assert added_obj.operation == "FORGET"
    assert added_obj.sequence_number == 1  # MAX(0) + 1
    assert added_obj.expires_at is not None
    payload = json.loads(added_obj.payload)
    assert payload["node_slugs"] == node_slugs
    assert payload["edge_slugs"] == edge_slugs


@pytest.mark.asyncio
async def test_log_version_event_sequence_increments():
    """Sequence number should be max + 1."""
    dataset_id = uuid4()
    session_mock = _make_session_mock()
    # Pretend max_seq is 5
    session_mock.execute.return_value.scalar_one.return_value = 5

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event("ADD", dataset_id, session=session_mock)

    added_obj = session_mock.add.call_args[0][0]
    assert added_obj.sequence_number == 6


@pytest.mark.asyncio
async def test_log_version_event_expires_at_defaults_30_days():
    dataset_id = uuid4()
    session_mock = _make_session_mock()

    before = datetime.now(timezone.utc)

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event("FORGET", dataset_id, session=session_mock)

    added_obj = session_mock.add.call_args[0][0]
    delta = added_obj.expires_at - before
    # Should be approximately 30 days
    assert timedelta(days=29) < delta < timedelta(days=31)


@pytest.mark.asyncio
async def test_log_version_event_empty_slugs():
    """Logging with no slugs stores empty lists."""
    dataset_id = uuid4()
    session_mock = _make_session_mock()

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event("ADD", dataset_id, session=session_mock)

    added_obj = session_mock.add.call_args[0][0]
    payload = json.loads(added_obj.payload)
    assert payload["node_slugs"] == []
    assert payload["edge_slugs"] == []
    assert payload["datapoints"] == []


@pytest.mark.asyncio
async def test_log_version_event_captures_datapoint_snapshots():
    """ADD event should carry DataPoint JSON snapshots in payload."""
    dataset_id = uuid4()
    node_id = str(uuid4())
    snapshot_json = json.dumps({"id": node_id, "type": "Entity", "name": "Paris"})

    session_mock = _make_session_mock()

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event(
            "ADD",
            dataset_id,
            node_slugs=[node_id],
            datapoint_snapshots=[snapshot_json],
            session=session_mock,
        )

    added_obj = session_mock.add.call_args[0][0]
    payload = json.loads(added_obj.payload)
    assert len(payload["datapoints"]) == 1
    assert json.loads(payload["datapoints"][0])["name"] == "Paris"
