from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, Text, DateTime
from cognee.infrastructure.databases.relational import Base, UUID
from .DatasetData import DatasetData

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID, primary_key = True, default = uuid4)

    name = Column(Text)

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone = True), onupdate = lambda: datetime.now(timezone.utc))

    owner_id = Column(UUID, index = True)

    data: Mapped[List["Data"]] = relationship(
        secondary = DatasetData.__tablename__,
        back_populates = "datasets",
        lazy = "noload",
    )

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "ownerId": str(self.owner_id),
            "data": [data.to_json() for data in self.data]
        }
