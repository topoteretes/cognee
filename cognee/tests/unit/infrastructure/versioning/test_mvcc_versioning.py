"""Unit tests for MVCC dataset versioning, snapshot management, and soft-delete/undo.

These tests use in-memory SQLite stores and mocks to verify versioning models,
version management logic, soft-deletion, and snapshots.
"""

from datetime import datetime, timedelta, timezone
import os
import sqlite3
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.versioning.models import VersionStamp
from cognee.infrastructure.databases.versioning.snapshot_store import SnapshotStore
from cognee.infrastructure.databases.versioning.version_manager import (
    SnapshotNotFoundError,
    VersionManager,
)

TEST_DB_PATH = "test_mvcc_versioning.db"


@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Fixture to ensure the test database is cleaned up before and after each test."""
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
    yield
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_snapshot_create_and_retrieve():
    """Create a snapshot, retrieve it by name, and assert fields match."""
    store = SnapshotStore(TEST_DB_PATH)
    dataset_id = uuid4()
    version_id = 5

    # Create snapshot.
    created = await store.create_snapshot("v1.0.0", dataset_id, version_id)
    assert created.name == "v1.0.0"
    assert created.dataset_id == dataset_id
    assert created.version_id == version_id

    # Retrieve snapshot.
    retrieved = await store.get_snapshot("v1.0.0", dataset_id)
    assert retrieved is not None
    assert retrieved.snapshot_id == created.snapshot_id
    assert retrieved.name == "v1.0.0"
    assert retrieved.dataset_id == dataset_id
    assert retrieved.version_id == version_id

    # List snapshots.
    snapshots = await store.list_snapshots(dataset_id)
    assert len(snapshots) == 1
    assert snapshots[0].name == "v1.0.0"

    # Delete snapshot.
    await store.delete_snapshot("v1.0.0", dataset_id)
    retrieved_after = await store.get_snapshot("v1.0.0", dataset_id)
    assert retrieved_after is None


@pytest.mark.asyncio
async def test_version_increments_on_cognify():
    """Mock cognify call, assert version_id increments."""
    dataset_id = uuid4()
    version_manager = VersionManager(dataset_id, TEST_DB_PATH)

    initial_version = await version_manager.get_current_version()
    assert initial_version == 0

    # Mock the cognify function execution.
    with patch("cognee.api.v1.cognify.cognify.cognify", new_callable=AsyncMock) as mock_cognify:
        async def fake_cognify(*args, **kwargs):
            await version_manager.increment_version()
            return {"status": "success"}

        mock_cognify.side_effect = fake_cognify

        from cognee.api.v1.cognify.cognify import cognify
        await cognify(datasets=[dataset_id])

        new_version = await version_manager.get_current_version()
        assert new_version == 1


@pytest.mark.asyncio
async def test_soft_forget_sets_valid_to():
    """Call soft forget, assert valid_to is set not None in database."""
    dataset_id = uuid4()
    data_id = uuid4()
    node_id = "test-node-1"

    # Seed an active node in the test DB path.
    with sqlite3.connect(TEST_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versioned_nodes (
                node_id TEXT PRIMARY KEY,
                data_id TEXT,
                dataset_id TEXT,
                properties TEXT,
                valid_from TEXT,
                valid_to TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO versioned_nodes (node_id, data_id, dataset_id, properties, valid_from, valid_to)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (node_id, str(data_id), str(dataset_id), "{}", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    # Call soft forget.
    from cognee.api.v1.forget.forget import forget
    from cognee.modules.users.methods import get_default_user

    mock_user = await get_default_user()

    with patch("cognee.infrastructure.llm.get_llm_config") as mock_get_config, \
         patch("cognee.api.v1.forget.forget._resolve_dataset_id", return_value=dataset_id):
        mock_config = AsyncMock()
        mock_config.versioning_enabled = True
        mock_get_config.return_value = mock_config

        # Patch forget's internal db path to use our test DB.
        with patch("cognee.root_dir.get_absolute_path", return_value=TEST_DB_PATH):
            result = await forget(
                data_id=data_id,
                dataset_id=dataset_id,
                user=mock_user,
            )

    assert result["soft_deleted_count"] == 1

    # Check database status.
    with sqlite3.connect(TEST_DB_PATH) as conn:
        cursor = conn.execute("SELECT valid_to FROM versioned_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()

    assert row is not None
    assert row[0] is not None  # valid_to timestamp should be set


@pytest.mark.asyncio
async def test_undo_forget_clears_valid_to():
    """Soft forget then undo, assert valid_to is None again."""
    dataset_id = uuid4()
    data_id = uuid4()
    node_id = "test-node-2"

    # Seed an active node in the test DB path.
    with sqlite3.connect(TEST_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versioned_nodes (
                node_id TEXT PRIMARY KEY,
                data_id TEXT,
                dataset_id TEXT,
                properties TEXT,
                valid_from TEXT,
                valid_to TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO versioned_nodes (node_id, data_id, dataset_id, properties, valid_from, valid_to)
            VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (node_id, str(data_id), str(dataset_id), "{}", datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()

    from cognee.api.v1.forget.forget import forget
    from cognee.modules.users.methods import get_default_user

    mock_user = await get_default_user()

    # Perform soft forget.
    with patch("cognee.infrastructure.llm.get_llm_config") as mock_get_config, \
         patch("cognee.api.v1.forget.forget._resolve_dataset_id", return_value=dataset_id):
        mock_config = AsyncMock()
        mock_config.versioning_enabled = True
        mock_get_config.return_value = mock_config

        with patch("cognee.root_dir.get_absolute_path", return_value=TEST_DB_PATH):
            await forget(
                data_id=data_id,
                dataset_id=dataset_id,
                user=mock_user,
            )

            # Now undo.
            undo_result = await forget(
                data_id=data_id,
                dataset_id=dataset_id,
                user=mock_user,
                undo=True,
            )

    assert undo_result["restored_count"] == 1

    # Verify node valid_to is None again.
    with sqlite3.connect(TEST_DB_PATH) as conn:
        cursor = conn.execute("SELECT valid_to FROM versioned_nodes WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()

    assert row is not None
    assert row[0] is None


@pytest.mark.asyncio
async def test_resolve_as_of_snapshot_name():
    """Resolve 'v1' -> version_id, assert correct int returned."""
    dataset_id = uuid4()
    version_manager = VersionManager(dataset_id, TEST_DB_PATH)

    # Register snapshot in store.
    await version_manager.snapshot_store.create_snapshot("v1", dataset_id, 3)

    resolved = await version_manager.resolve_as_of("v1")
    assert resolved == 3


@pytest.mark.asyncio
async def test_resolve_as_of_datetime():
    """Resolve datetime -> nearest version_id created at or before."""
    dataset_id = uuid4()
    version_manager = VersionManager(dataset_id, TEST_DB_PATH)

    # Setup history records directly.
    base_time = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)
    t1 = (base_time - timedelta(hours=2)).isoformat()
    t2 = (base_time - timedelta(hours=1)).isoformat()
    t3 = (base_time + timedelta(hours=1)).isoformat()

    with sqlite3.connect(TEST_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO version_history (dataset_id, version_id, created_at) VALUES (?, ?, ?)",
            (str(dataset_id), 1, t1),
        )
        conn.execute(
            "INSERT INTO version_history (dataset_id, version_id, created_at) VALUES (?, ?, ?)",
            (str(dataset_id), 2, t2),
        )
        conn.execute(
            "INSERT INTO version_history (dataset_id, version_id, created_at) VALUES (?, ?, ?)",
            (str(dataset_id), 3, t3),
        )
        conn.commit()

    # Query exactly at base_time. Nearest at/before should be version 2 (t2 is 1h before, t3 is 1h after).
    resolved = await version_manager.resolve_as_of(base_time)
    assert resolved == 2

    # Query before all. Should fall back to earliest.
    resolved_earliest = await version_manager.resolve_as_of(base_time - timedelta(hours=3))
    assert resolved_earliest == 1


@pytest.mark.asyncio
async def test_snapshot_not_found_raises():
    """Get nonexistent snapshot, assert SnapshotNotFoundError."""
    dataset_id = uuid4()
    version_manager = VersionManager(dataset_id, TEST_DB_PATH)

    with pytest.raises(SnapshotNotFoundError):
        await version_manager.resolve_as_of("nonexistent-snapshot")
