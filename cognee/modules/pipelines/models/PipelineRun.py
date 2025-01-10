import enum
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, JSON, Enum, UUID
from cognee.infrastructure.databases.relational import Base


class PipelineRunStatus(enum.Enum):
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    status = Column(Enum(PipelineRunStatus))

    run_id = Column(UUID, index=True)
    run_info = Column(JSON)
