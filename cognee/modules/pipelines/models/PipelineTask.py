from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, UUID, ForeignKey
from cognee.infrastructure.databases.relational import ModelBase

class PipelineTask(ModelBase):
    __tablename__ = "pipeline_task"

    id = Column(UUID, primary_key = True, default = uuid4())

    created_at = Column(DateTime, default = datetime.now(timezone.utc))

    pipeline_id = Column("pipeline", UUID, ForeignKey("pipeline.id"), primary_key = True)
    task_id = Column("task", UUID, ForeignKey("task.id"), primary_key = True)
