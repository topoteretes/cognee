import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, JSON, Enum, UUID, String, Index
from cognee.infrastructure.databases.relational import Base


class PipelineRunStatus(enum.Enum):
    DATASET_PROCESSING_INITIATED = "DATASET_PROCESSING_INITIATED"
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"


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

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    status = Column(Enum(PipelineRunStatus))
    pipeline_run_id = Column(UUID, index=True)
    pipeline_name = Column(String)
    pipeline_id = Column(UUID, index=True)
    dataset_id = Column(UUID, index=True)
    run_info = Column(JSON)
