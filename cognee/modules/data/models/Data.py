from datetime import datetime, timezone
from typing import List
from uuid import uuid4
from sqlalchemy import UUID, Column, DateTime, String
from sqlalchemy.orm import Mapped, relationship

from cognee.infrastructure.databases.relational import Base

from .DatasetData import DatasetData
from .Metadata import Metadata


class Data(Base):
    __tablename__ = "data"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(String)
    extension = Column(String)
    mime_type = Column(String)
    raw_data_location = Column(String)
    owner_id = Column(UUID, index=True)
    content_hash = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    datasets = relationship(
        "Dataset",
        secondary=DatasetData.__tablename__,
        back_populates="data",
        lazy="noload",
        cascade="all, delete",
    )

    metadata_relationship = relationship(
        "Metadata",
        back_populates="data",
        lazy="noload",
        cascade="all, delete",
    )

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "extension": self.extension,
            "mimeType": self.mime_type,
            "rawDataLocation": self.raw_data_location,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            # "datasets": [dataset.to_json() for dataset in self.datasets]
        }
