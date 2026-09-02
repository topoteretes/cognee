import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Enum, Index, Integer, JSON, String, UUID
from cognee.infrastructure.databases.relational import Base


class PipelineRunStatus(enum.Enum):
    DATASET_PROCESSING_INITIATED = "DATASET_PROCESSING_INITIATED"
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"


class OperationOutcome(str, enum.Enum):
    """Operation-level result, orthogonal to the dataset-shaped PipelineRunStatus.

    Stored as a plain String column (via ``.value``), NOT a DB enum — the
    PipelineRunStatus enum is a public API response model and a native
    Postgres enum type, so it stays frozen.
    """

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        # Covers get_pipeline_status.py / get_pipeline_progress.py's
        # ROW_NUMBER() lookup of each dataset's latest run for a pipeline
        # (filter on dataset_id + pipeline_name, order by created_at DESC).
        # See alembic/versions/d1e2f3a4b5c6_add_pipeline_runs_status_index.py
        # for the migration that adds this to existing databases.
        Index(
            "ix_pipeline_runs_dataset_pipeline_created_at",
            "dataset_id",
            "pipeline_name",
            "created_at",
        ),
    )

    # Readers of this table page newest-first with id as the tiebreaker
    # (ORDER BY created_at DESC, id DESC) or range-scan a created_at window,
    # so the two columns are indexed together. Composite, not created_at
    # alone: without id, OFFSET paging re-serves rows sharing a timestamp.
    # Mirrored by migration c4e8a1f6b3d7 for databases created before it.
    __table_args__ = (Index("ix_pipeline_runs_created_at_id", "created_at", "id"),)

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    status = Column(Enum(PipelineRunStatus))
    pipeline_run_id = Column(UUID, index=True)
    pipeline_name = Column(String)
    pipeline_id = Column(UUID, index=True)
    dataset_id = Column(UUID, index=True)
    run_info = Column(JSON)

    # Operation-record columns (SDK-399). All nullable: rows written before
    # this change stay NULL — no backfill. Non-pipeline operations (search,
    # recall, forget, remember, delete, prune) write exactly one row with
    # status/pipeline_name/pipeline_id NULL, which keeps them invisible to
    # the legacy latest-row status readers.
    user_id = Column(UUID, index=True)
    tenant_id = Column(UUID)
    operation_name = Column(String, index=True)
    started_at = Column(DateTime(timezone=True))
    ended_at = Column(DateTime(timezone=True))
    outcome = Column(String, index=True)  # OperationOutcome values: "succeeded" / "failed"
    error_class = Column(String)  # exception class name, e.g. "DatasetNotFoundError"
    tokens_in = Column(Integer)  # NULL = not measured; 0 = measured zero
    tokens_out = Column(Integer)
    origin = Column(String)  # initiating surface: "sdk" / "api" / "cli" / "mcp" / "background"
    session_id = Column(String, index=True)  # session-cache id; joins SessionModelUsage
    # Parent operation's pipeline_run_id — makes remember → add/cognify/improve
    # nesting a queryable tree (token totals chain to parents; never SUM across levels).
    parent_operation_id = Column(UUID, index=True)
    # True = launched background work, so outcome="succeeded" means "accepted
    # and started", not "background work finished". NULL = not applicable.
    background = Column(Boolean)
    error_message = Column(String)  # PII-scrubbed, truncated (see operations/scrub_error.py)
