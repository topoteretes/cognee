from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import UUID, Column, DateTime, String, JSON, Integer
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base

from .DatasetData import DatasetData


class Data(Base):
    __tablename__ = "data"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(String)
    extension = Column(String)
    mime_type = Column(String)
    original_extension = Column(String, nullable=True)
    original_mime_type = Column(String, nullable=True)
    loader_engine = Column(String)
    raw_data_location = Column(String)
    original_data_location = Column(String)
    owner_id = Column(UUID, index=True)
    tenant_id = Column(UUID, index=True, nullable=True)
    content_hash = Column(String)
    raw_content_hash = Column(String)
    external_metadata = Column(JSON)
    # Store NodeSet as JSON list of strings
    node_set = Column(JSON, nullable=True)
    # MutableDict allows SQLAlchemy to notice key-value pair changes, without it changing a value for a key
    # wouldn't be noticed when commiting a database session
    pipeline_status = Column(MutableDict.as_mutable(JSON))
    token_count = Column(Integer)
    data_size = Column(Integer, nullable=True)  # File size in bytes
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    datasets = relationship(
        "Dataset",
        secondary=DatasetData.__tablename__,
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
            "nodeSet": self.node_set,
            # "datasets": [dataset.to_json() for dataset in self.datasets]
        }
