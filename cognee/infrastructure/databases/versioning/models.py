"""Multiversion Concurrency Control (MVCC) dataset versioning models.

This module defines the core data structures used to stamp and represent versions of
data points, nodes, and edges under an MVCC system.

MVCC Design:
~~~~~~~~~~~~
In this design, records are never mutated or deleted in place. Instead:
- Every insert or update stamps the item with a `valid_from` timestamp and `version_id`.
- The `valid_to` field denotes when a record was superseded or deleted.
- A value of `valid_to = None` signifies that the record is "currently live" or the active head.
- When an item is modified, the active record gets its `valid_to` set to the current transaction
  time, and a new record with `valid_from = current_time` and `valid_to = None` is inserted.
- When an item is deleted, its active record gets its `valid_to` set to the current time (soft-delete/tombstoned),
  and no new record is created.
- This allows snapshot queries ("as of" a specific time or version ID) by filtering for:
  `valid_from <= target_time AND (valid_to > target_time OR valid_to IS None)`
"""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass
class VersionStamp:
    """Stamps a data point, node, or edge with its MVCC lifespan metadata.

    Attributes:
        version_id: Monotonic integer version identifier for the dataset.
        valid_from: The datetime when this version of the record became active.
        valid_to: The datetime when this version was superseded or soft-deleted.
            A value of None indicates the record is currently active/live.
    """

    version_id: int
    valid_from: datetime
    valid_to: datetime | None = None


@dataclass
class SnapshotPointer:
    """Represents a named pointer to a specific version of a dataset (a snapshot).

    Attributes:
        snapshot_id: Unique UUID identifier for the snapshot.
        name: User-defined tag or name for the snapshot (e.g. "v1.0.0").
        dataset_id: The ID of the dataset this snapshot belongs to.
        version_id: The dataset version ID that this snapshot points to.
        created_at: The datetime when the snapshot pointer was created.
    """

    snapshot_id: UUID
    name: str
    dataset_id: UUID
    version_id: int
    created_at: datetime


@dataclass
class VersioningConfig:
    """Configuration settings for MVCC retention and checkpointing.

    Attributes:
        retention_days: Number of days to keep historical versions before pruning.
            Default is 30.
        checkpoint_interval: Number of versions/operations between database checkpoints.
            Default is 50.
    """

    retention_days: int = 30
    checkpoint_interval: int = 50
