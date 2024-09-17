from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, JSON
from cognee.infrastructure.databases.relational import Base, UUID

class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(UUID, primary_key = True, default = uuid4)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))

    status = Column(String)

    run_id = Column(UUID, index = True)
    run_info = Column(JSON)
