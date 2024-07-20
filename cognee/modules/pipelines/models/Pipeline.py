from typing import List
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, UUID, DateTime, String, Text
from sqlalchemy.orm import relationship, Mapped
from cognee.infrastructure.databases.relational import ModelBase
from .PipelineTask import PipelineTask

class Pipeline(ModelBase):
    __tablename__ = "pipelines"

    id = Column(UUID, primary_key = True, default = uuid4())
    name = Column(String)
    description = Column(Text, nullable = True)

    created_at = Column(DateTime, default = datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate = datetime.now(timezone.utc))

    tasks = Mapped[List["Task"]] = relationship(
        secondary = PipelineTask.__tablename__,
        back_populates = "pipeline",
    )
