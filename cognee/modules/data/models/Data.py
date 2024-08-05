from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, String, DateTime, UUID
from cognee.infrastructure.databases.relational import Base
from .DatasetData import DatasetData

class Data(Base):
    __tablename__ = "data"

    id = Column(UUID(as_uuid = True), primary_key = True, default = uuid4)

    name = Column(String)
    extension = Column(String)
    mime_type = Column(String)
    raw_data_location = Column(String)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone = True), onupdate = lambda: datetime.now(timezone.utc))

    datasets: Mapped[List["Dataset"]] = relationship(
        secondary = DatasetData.__tablename__,
        back_populates = "data"
    )
