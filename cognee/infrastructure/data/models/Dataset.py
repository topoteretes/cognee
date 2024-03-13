from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, Text, DateTime, UUID
from cognee.infrastructure.databases.relational import ModelBase
from .DatasetData import DatasetData

class Dataset(ModelBase):
    __tablename__ = "dataset"

    id = Column(UUID, primary_key = True)
    name = Column(Text)
    description = Column(Text, nullable = True)

    created_at = Column(DateTime, default = datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate = datetime.now(timezone.utc))

    data: Mapped[List["Data"]] = relationship(
        secondary = DatasetData.__tablename__,
        back_populates = "datasets"
    )
