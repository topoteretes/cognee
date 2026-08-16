from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, DateTime, Text, Uuid
from sqlalchemy.types import JSON as GenericJSON
from cognee.infrastructure.databases.relational import Base


class DatasetConfiguration(Base):
    __tablename__ = "dataset_configurations"

    id = Column(Uuid, primary_key=True, default=uuid4)
    dataset_id = Column(
        Uuid, ForeignKey("datasets.id", ondelete="CASCADE"), unique=True, nullable=False
    )

    graph_schema = Column(GenericJSON, nullable=True)
    custom_prompt = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    dataset = relationship("Dataset", back_populates="configuration")
