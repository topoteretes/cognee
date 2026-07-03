"""Version manager for MVCC dataset versioning.

This module provides the VersionManager class, which tracks and manages versions
for each dataset. It supports version incrementing and "as of" resolution.
"""

from datetime import datetime, timezone
from functools import lru_cache
import sqlite3
from typing import Optional, Union
from uuid import UUID

from cognee.infrastructure.databases.versioning.snapshot_store import SnapshotStore
from cognee.shared.logging_utils import get_logger

logger = get_logger("version_manager")


class SnapshotNotFoundError(Exception):
    """Raised when a requested snapshot is not found in the store."""

    def __init__(self, name: str, dataset_id: UUID):
        self.name = name
        self.dataset_id = dataset_id
        super().__init__(f"Snapshot '{name}' not found for dataset '{dataset_id}'")


class VersionManager:
    """Manages version tracking and resolution for a specific dataset.

    Persists version counters and timestamps in a SQLite database.
    """

    def __init__(self, dataset_id: UUID, db_path: str = ":memory:"):
        """Initialise the VersionManager.

        Args:
            dataset_id: The ID of the dataset to manage.
            db_path: Path to the SQLite database file.
        """
        self.dataset_id = dataset_id
        self.db_path = db_path
        self.snapshot_store = SnapshotStore(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Create the dataset_versions and version_history tables if they do not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_id TEXT PRIMARY KEY,
                    current_version_id INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS version_history (
                    dataset_id TEXT NOT NULL,
                    version_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, version_id)
                )
                """
            )
            # Insert default row for dataset if it does not exist.
            conn.execute(
                """
                INSERT OR IGNORE INTO dataset_versions (dataset_id, current_version_id)
                VALUES (?, 0)
                """,
                (str(self.dataset_id),),
            )
            conn.commit()

    async def get_current_version(self) -> int:
        """Retrieve the current version ID of the dataset.

        Returns:
            The current active version ID (integer).
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT current_version_id FROM dataset_versions WHERE dataset_id = ?",
                (str(self.dataset_id),),
            )
            row = cursor.fetchone()

        return row[0] if row else 0

    async def increment_version(self) -> int:
        """Increment and return the dataset version.

        Stamps the new version with the current UTC timestamp in the version history.

        Returns:
            The newly incremented version ID.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            # Increment current_version_id.
            conn.execute(
                """
                UPDATE dataset_versions
                SET current_version_id = current_version_id + 1
                WHERE dataset_id = ?
                """,
                (str(self.dataset_id),),
            )
            # Retrieve the incremented value.
            cursor = conn.execute(
                "SELECT current_version_id FROM dataset_versions WHERE dataset_id = ?",
                (str(self.dataset_id),),
            )
            new_version = cursor.fetchone()[0]

            # Record in version history.
            conn.execute(
                """
                INSERT INTO version_history (dataset_id, version_id, created_at)
                VALUES (?, ?, ?)
                """,
                (str(self.dataset_id), new_version, created_at),
            )
            conn.commit()

        logger.info("Dataset %s version incremented to %d", self.dataset_id, new_version)
        return new_version

    async def resolve_as_of(self, as_of: Optional[Union[str, datetime]] = None) -> Optional[int]:
        """Resolve an 'as_of' parameter into a specific version ID.

        Args:
            as_of: The version parameter to resolve:
                - None: returns the latest version ID.
                - str: resolves as a snapshot name.
                - datetime: resolves as the nearest version ID created at or before the datetime.

        Returns:
            The resolved version ID (int) or None if it cannot be resolved.

        Raises:
            SnapshotNotFoundError: If as_of is a string and the snapshot does not exist.
        """
        if as_of is None:
            return await self.get_current_version()

        if isinstance(as_of, str):
            snapshot = await self.snapshot_store.get_snapshot(as_of, self.dataset_id)
            if snapshot is None:
                raise SnapshotNotFoundError(as_of, self.dataset_id)
            return snapshot.version_id

        if isinstance(as_of, datetime):
            # Resolve to the nearest version created at or before the given datetime.
            target_iso = as_of.isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id FROM version_history
                    WHERE dataset_id = ? AND created_at <= ?
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (str(self.dataset_id), target_iso),
                )
                row = cursor.fetchone()
            if row:
                return row[0]
            # Fall back to the earliest recorded version if no match is found before the datetime.
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT version_id FROM version_history
                    WHERE dataset_id = ?
                    ORDER BY created_at ASC LIMIT 1
                    """,
                    (str(self.dataset_id),),
                )
                row = cursor.fetchone()
            return row[0] if row else 0

        raise ValueError(f"Invalid as_of parameter type: {type(as_of)}")


@lru_cache(maxsize=128)
def get_version_manager(dataset_id: UUID, db_path: str = ":memory:") -> VersionManager:
    """Retrieve or create a cached VersionManager instance for the given dataset ID."""
    return VersionManager(dataset_id, db_path)
