"""Unit tests for Checkpoint model and create_checkpoint operation."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Checkpoint model — metadata only
# ---------------------------------------------------------------------------


def test_checkpoint_tablename():
    from cognee.modules.versioning.models.Checkpoint import Checkpoint

    assert Checkpoint.__tablename__ == "versioning_checkpoints"


def test_checkpoint_has_required_columns():
    from cognee.modules.versioning.models.Checkpoint import Checkpoint

    cols = {c.name for c in Checkpoint.__table__.columns}
    for required in ("id", "dataset_id", "user_id", "label", "created_at", "node_slugs", "edge_slugs"):
        assert required in cols, f"Missing column: {required}"


def test_checkpoint_has_dataset_index():
    from cognee.modules.versioning.models.Checkpoint import Checkpoint

    index_names = {idx.name for idx in Checkpoint.__table__.indexes}
    assert "idx_versioning_checkpoints_dataset_id" in index_names


# ---------------------------------------------------------------------------
# create_checkpoint — mocked DB using plain MagicMock ledger rows
# ---------------------------------------------------------------------------


def _make_ledger_row(src_id=None, dst_id=None, label="relates_to"):
    """Return a MagicMock that mimics a GraphRelationshipLedger row."""
    row = MagicMock()
    row.source_node_id = src_id if src_id is not None else uuid4()
    row.destination_node_id = dst_id if dst_id is not None else uuid4()
    row.node_label = label
    row.deleted_at = None
    return row


def _make_session(rows):
    scalars_mock = MagicMock()
    scalars_mock.all.return_value = rows
    execute_result = MagicMock()
    execute_result.scalars.return_value = scalars_mock

    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_create_checkpoint_snapshots_alive_rows():
    dataset_id = uuid4()
    row_a = _make_ledger_row(label="A")
    row_b = _make_ledger_row(label="B")
    session = _make_session([row_a, row_b])

    with patch(
        "cognee.modules.versioning.operations.create_checkpoint.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.create_checkpoint import create_checkpoint

        await create_checkpoint(dataset_id, label="v1", session=session)

    session.add.assert_called_once()
    added = session.add.call_args[0][0]
    node_slugs = json.loads(added.node_slugs)
    edge_slugs = json.loads(added.edge_slugs)

    # Four unique node IDs across two rows (src_a, dst_a, src_b, dst_b)
    assert len(node_slugs) == 4
    assert "A" in edge_slugs
    assert "B" in edge_slugs
    assert added.label == "v1"


@pytest.mark.asyncio
async def test_create_checkpoint_empty_ledger():
    """An empty ledger should produce a checkpoint with empty lists."""
    dataset_id = uuid4()
    session = _make_session([])

    with patch(
        "cognee.modules.versioning.operations.create_checkpoint.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.create_checkpoint import create_checkpoint

        await create_checkpoint(dataset_id, session=session)

    added = session.add.call_args[0][0]
    assert json.loads(added.node_slugs) == []
    assert json.loads(added.edge_slugs) == []


@pytest.mark.asyncio
async def test_create_checkpoint_deduplicates_node_slugs():
    """If a node appears as source and destination it should only appear once."""
    dataset_id = uuid4()
    shared_id = uuid4()
    row = _make_ledger_row(src_id=shared_id, dst_id=shared_id, label="self_link")
    session = _make_session([row, row])  # two rows sharing the same node ID

    with patch(
        "cognee.modules.versioning.operations.create_checkpoint.with_async_session",
        lambda fn: fn,
    ):
        from cognee.modules.versioning.operations.create_checkpoint import create_checkpoint

        await create_checkpoint(dataset_id, session=session)

    added = session.add.call_args[0][0]
    node_slugs = json.loads(added.node_slugs)
    assert len(node_slugs) == 1
    assert str(shared_id) in node_slugs
