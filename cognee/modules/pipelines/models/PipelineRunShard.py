"""The shards of a pipeline run whose data items go to workers.

Such a run (``cognee.modules.pipelines.execution``) cuts its items into shards,
one per worker. A row per shard and a row per item of a shard are how a shard
reaches its worker and how the worker's answer comes back, and they stay as the
record of what the run handed out and how it ended.

The process that owns the run writes the rows. A shard's worker takes its row,
and stores on it and on its item rows how the items ended. What the items
produce is never stored here: a worker writes that to the stores itself.
"""

import enum
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Uuid

from cognee.infrastructure.databases.relational import Base


class ShardStatus(str, enum.Enum):
    """Where a shard stands. Its row holds the value."""

    # Handed out; no worker has taken it yet.
    PENDING = "pending"
    # A worker has taken it. A shard is taken once.
    RUNNING = "running"
    # Its worker stored how each of its items ended.
    FINISHED = "finished"
    # Its worker could not run it, and stored why (``error_class``,
    # ``error_message``). Its data rows are recorded as their completion stamps
    # had them; an item that is no row has no stamp and is recorded errored.
    FAILED = "failed"
    # Its worker ended without storing anything, or never took it. Its items
    # are recorded as for a failed shard.
    LOST = "lost"
    # The run ended while the shard was pending or running: it failed or was
    # cancelled, or its process died and the run was recovered. The shard's
    # items stay pending, whatever its worker had finished: the completion
    # stamp on each data row says that.
    INTERRUPTED = "interrupted"


class ShardItemStatus(str, enum.Enum):
    """How an item of a shard ended. Its row holds the value."""

    # Not answered for (yet).
    PENDING = "pending"
    COMPLETED = "completed"
    # The item was complete before the worker reached it.
    ALREADY_COMPLETED = "already_completed"
    ERRORED = "errored"


class PipelineRunShard(Base):
    """One shard of a pipeline run: the items one worker is given."""

    __tablename__ = "pipeline_run_shards"

    id = Column(Uuid, primary_key=True, default=uuid4)
    # The run. pipeline_runs has several rows per run, which all carry this id,
    # so there is no key a foreign key could reference.
    pipeline_run_id = Column(Uuid, index=True, nullable=False)
    dataset_id = Column(
        Uuid, ForeignKey("datasets.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # Position among the run's shards, from 0.
    shard_index = Column(Integer, nullable=False)
    # A ``ShardStatus`` value.
    status = Column(String(16), nullable=False)
    # The run as the worker needs it described (a ``PipelineRunSpec``): its
    # settings and how to find its dataset, its user and its tasks. Which items
    # the shard holds is in its item rows.
    spec = Column(JSON, nullable=False)
    # What the worker reported using. NULL unless the shard finished.
    tokens_in = Column(Integer)
    tokens_out = Column(Integer)
    embedding_tokens = Column(Integer)
    # Why the worker could not run the shard (``FAILED``): the class name and
    # the scrubbed message of what stopped it.
    error_class = Column(String)
    error_message = Column(String)
    # When the run handed its shards out: all of a run's shards at once, before
    # the first worker starts.
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    # When a worker took the shard, and when that worker stored its answer.
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))


class PipelineRunShardItem(Base):
    """One item of a shard, and how it ended.

    An item is a row of ``data`` (``data_id``: what it is, its name and where it
    is stored, is on that row) or a value the run was given that is no row
    (``value``).
    """

    __tablename__ = "pipeline_run_shard_items"

    shard_id = Column(
        Uuid, ForeignKey("pipeline_run_shards.id", ondelete="CASCADE"), primary_key=True
    )
    # Position among the shard's items, from 0.
    item_index = Column(Integer, primary_key=True)
    # The ``data`` row the item is; NULL for an item that is a value.
    data_id = Column(Uuid, ForeignKey("data.id", ondelete="CASCADE"), index=True)
    # The item itself, when it is no ``data`` row.
    value = Column(JSON)
    # A ``ShardItemStatus`` value.
    status = Column(String(32), nullable=False)
    # Set for an item that failed: the same class name and scrubbed message the
    # run reports for it (as on ``pipeline_runs``).
    error_class = Column(String)
    error_message = Column(String)
