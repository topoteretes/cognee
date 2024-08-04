from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, UUID, DateTime, String, JSON
from cognee.infrastructure.databases.relational import Base

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID, primary_key = True, default = uuid4)

    dataset_name = Column(String)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))

    run_info = Column(JSON)
