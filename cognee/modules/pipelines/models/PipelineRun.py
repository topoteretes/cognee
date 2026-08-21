import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, JSON, Enum, Integer, UUID, String
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
