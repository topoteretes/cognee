"""Unit tests for VersionEvent model and log_version_event operation."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Helpers — use plain MagicMock so SQLAlchemy ORM state is not required
# ---------------------------------------------------------------------------


def _make_session_mock():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _event_stub(**kwargs):
    ev = MagicMock()
    ev.id = kwargs.get("id", uuid4())
    ev.operation = kwargs.get("operation", "FORGET")
    ev.dataset_id = kwargs.get("dataset_id", uuid4())
    ev.data_id = kwargs.get("data_id", uuid4())
    ev.user_id = kwargs.get("user_id", None)
    ev.run_id = kwargs.get("run_id", None)
    ev.payload = kwargs.get("payload", json.dumps({"node_slugs": [], "edge_slugs": []}))
    ev.created_at = None
    ev.undone_at = None
    return ev


# ---------------------------------------------------------------------------
# VersionEvent model — metadata only (no ORM instantiation)
# ---------------------------------------------------------------------------


def test_version_event_tablename():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    assert VersionEvent.__tablename__ == "version_events"


def test_version_event_has_required_columns():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    cols = {c.name for c in VersionEvent.__table__.columns}
    for required in ("id", "operation", "dataset_id", "data_id", "created_at", "undone_at", "payload"):
        assert required in cols, f"Missing column: {required}"


def test_version_event_has_indexes():
    from cognee.modules.versioning.models.VersionEvent import VersionEvent

    index_names = {idx.name for idx in VersionEvent.__table__.indexes}
    assert "idx_version_events_dataset_id" in index_names
    assert "idx_version_events_operation" in index_names


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
    payload = json.loads(added_obj.payload)
    assert payload["node_slugs"] == node_slugs
    assert payload["edge_slugs"] == edge_slugs


@pytest.mark.asyncio
async def test_log_version_event_empty_slugs():
    """Logging with no slugs should store empty lists, not null."""
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


@pytest.mark.asyncio
async def test_log_version_event_cognify_operation():
    dataset_id = uuid4()
    session_mock = _make_session_mock()

    with patch(
        "cognee.modules.versioning.operations.log_event.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.log_event import log_version_event

        await log_version_event("COGNIFY", dataset_id, session=session_mock)

    added_obj = session_mock.add.call_args[0][0]
    assert added_obj.operation == "COGNIFY"
