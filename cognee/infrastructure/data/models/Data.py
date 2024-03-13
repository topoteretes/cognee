from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, MappedColumn, Mapped
from sqlalchemy import Column, String, DateTime, UUID, Text, JSON
from cognee.infrastructure.databases.relational import ModelBase
from .DatasetData import DatasetData

class Data(ModelBase):
    __tablename__ = "data"

    id = Column(UUID, primary_key = True)
    name = Column(String, nullable = True)
    description = Column(Text, nullable = True)
    raw_data_location = Column(String)
    meta_data: Mapped[dict] = MappedColumn(type_ = JSON) # metadata is reserved word

    created_at = Column(DateTime, default = datetime.now(timezone.utc))
    updated_at = Column(DateTime, onupdate = datetime.now(timezone.utc))

    datasets: Mapped[List["Dataset"]] = relationship(
        secondary = DatasetData.__tablename__,
        back_populates = "data"
    )
