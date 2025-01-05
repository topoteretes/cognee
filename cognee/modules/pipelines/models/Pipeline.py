from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, Text, UUID
from sqlalchemy.orm import relationship, Mapped
from cognee.infrastructure.databases.relational import Base
from .PipelineTask import PipelineTask
from .Task import Task


class Pipeline(Base):
    __tablename__ = "pipelines"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(String)
    description = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    tasks = Mapped[list["Task"]] = relationship(
        secondary=PipelineTask.__tablename__,
        back_populates="pipeline",
    )
