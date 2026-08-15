from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Uuid
from cognee.infrastructure.databases.relational import Base


class PipelineTask(Base):
    __tablename__ = "pipeline_task"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    pipeline_id = Column("pipeline", Uuid, ForeignKey("pipeline.id"), primary_key=True)
    task_id = Column("task", Uuid, ForeignKey("task.id"), primary_key=True)
