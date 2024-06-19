from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, String, DateTime, UUID, Text
from cognee.infrastructure.databases.relational import ModelBase
from .PipelineTask import PipelineTask

class Task(ModelBase):
    __tablename__ = "tasks"

    id = Column(UUID, primary_key = True, default = uuid4())
    name = Column(String)
    description = Column(Text, nullable = True)

    executable = Column(Text)

    created_at = Column(DateTime, default = datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate = datetime.now(timezone.utc))

    datasets: Mapped[List["Pipeline"]] = relationship(
        secondary = PipelineTask.__tablename__,
        back_populates = "task"
    )
