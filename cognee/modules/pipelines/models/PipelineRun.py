import enum
from uuid import uuid4, UUID as UUID_t
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column, DateTime, JSON, Enum, UUID, String
from cognee.infrastructure.databases.relational import Base


class PipelineRunStatus(enum.Enum):
    DATASET_PROCESSING_INITIATED = "DATASET_PROCESSING_INITIATED"
    DATASET_PROCESSING_STARTED = "DATASET_PROCESSING_STARTED"
    DATASET_PROCESSING_COMPLETED = "DATASET_PROCESSING_COMPLETED"
    DATASET_PROCESSING_ERRORED = "DATASET_PROCESSING_ERRORED"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    status: Mapped[PipelineRunStatus] = mapped_column(Enum(PipelineRunStatus))
    pipeline_run_id: Mapped[UUID_t] = mapped_column(UUID, index=True)
    pipeline_name = Column(String)
    pipeline_id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), index=True)
    dataset_id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), index=True)
    run_info = Column(JSON)
