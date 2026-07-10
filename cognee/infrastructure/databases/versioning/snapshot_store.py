"""Snapshot store for MVCC dataset versioning.

This module provides the SnapshotStore class, which manages snapshot pointers for
datasets. It supports both in-memory and SQLite-backed storage.
"""

import sqlite3
from datetime import datetime, timezone
import os
from typing import List, Optional
from uuid import UUID, uuid4

from cognee.infrastructure.databases.versioning.models import SnapshotPointer
from cognee.shared.logging_utils import get_logger

logger = get_logger("snapshot_store")


class SnapshotStore:
    """Store for managing dataset snapshots.

    Supports in-memory storage (default) or persistence in a SQLite database file.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialise the SnapshotStore.

        Args:
            db_path: Path to the SQLite database file, or ':memory:' for in-memory storage.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the snapshots table if it does not exist."""
        # Ensure directories exist if db_path is a file path.
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    version_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(name, dataset_id)
                )
                """
            )
            conn.commit()

    async def create_snapshot(self, name: str, dataset_id: UUID, version_id: int) -> SnapshotPointer:
        """Create a new snapshot pointer for a dataset.

        Args:
            name: User-defined name for the snapshot.
            dataset_id: The ID of the dataset to snapshot.
            version_id: The version ID to associate with the snapshot.

        Returns:
            The created SnapshotPointer.

        Raises:
            ValueError: If a snapshot with the same name already exists for the dataset.
        """
        snapshot_id = uuid4()
        created_at = datetime.now(timezone.utc)

        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO snapshots (snapshot_id, name, dataset_id, version_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(snapshot_id),
                        name,
                        str(dataset_id),
                        version_id,
                        created_at.isoformat(),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"Snapshot with name '{name}' already exists for dataset '{dataset_id}'"
                ) from error

        logger.info(
            "Created snapshot: %s for dataset %s pointing to version %d",
            name,
            dataset_id,
            version_id,
        )

        return SnapshotPointer(
            snapshot_id=snapshot_id,
            name=name,
            dataset_id=dataset_id,
            version_id=version_id,
            created_at=created_at,
        )

    async def get_snapshot(self, name: str, dataset_id: UUID) -> Optional[SnapshotPointer]:
        """Retrieve a snapshot pointer by its name and dataset ID.

        Args:
            name: The name of the snapshot.
            dataset_id: The ID of the dataset.

        Returns:
            The SnapshotPointer if found, otherwise None.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT snapshot_id, name, dataset_id, version_id, created_at
                FROM snapshots
                WHERE name = ? AND dataset_id = ?
                """,
                (name, str(dataset_id)),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return SnapshotPointer(
            snapshot_id=UUID(row[0]),
            name=row[1],
            dataset_id=UUID(row[2]),
            version_id=row[3],
            created_at=datetime.fromisoformat(row[4]),
        )

    async def list_snapshots(self, dataset_id: UUID) -> List[SnapshotPointer]:
        """List all snapshots available for a dataset.

        Args:
            dataset_id: The ID of the dataset.

        Returns:
            A list of SnapshotPointers associated with the dataset.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT snapshot_id, name, dataset_id, version_id, created_at
                FROM snapshots
                WHERE dataset_id = ?
                ORDER BY created_at DESC
                """,
                (str(dataset_id),),
            )
            rows = cursor.fetchall()

        return [
            SnapshotPointer(
                snapshot_id=UUID(row[0]),
                name=row[1],
                dataset_id=UUID(row[2]),
                version_id=row[3],
                created_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        ]

    async def delete_snapshot(self, name: str, dataset_id: UUID) -> None:
        """Delete a snapshot pointer.

        Args:
            name: The name of the snapshot to delete.
            dataset_id: The ID of the dataset.

        Raises:
            ValueError: If the snapshot name does not exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM snapshots WHERE name = ? AND dataset_id = ?",
                (name, str(dataset_id)),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"Snapshot '{name}' not found for dataset '{dataset_id}'")

        logger.info("Deleted snapshot: %s for dataset %s", name, dataset_id)
