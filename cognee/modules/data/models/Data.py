from datetime import datetime, timezone
from uuid import uuid4
import enum
from dataclasses import dataclass
from sqlalchemy import UUID, Column, DateTime, String, JSON, Integer, Enum
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base

from .DatasetData import DatasetData


class FileProcessingStatus(enum.Enum):
    UNPROCESSED = "UNPROCESSED"
    PROCESSING = "PROCESSING"
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"


@dataclass
class ProcessingMetrics:
    """Metrics for file processing operations."""
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    processing_files: int = 0
    
    @property
    def completion_percentage(self) -> float:
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100


class Data(Base):
    __tablename__ = "data"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(String)
    extension = Column(String)
    mime_type = Column(String)
    raw_data_location = Column(String)
    owner_id = Column(UUID, index=True)
    content_hash = Column(String)
    external_metadata = Column(JSON)
    node_set = Column(JSON, nullable=True)  # Store NodeSet as JSON list of strings
    token_count = Column(Integer)
    processing_status = Column(Enum(FileProcessingStatus), default=FileProcessingStatus.UNPROCESSED)
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
            "processingStatus": self.processing_status.value if self.processing_status else None,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "nodeSet": self.node_set,
            # "datasets": [dataset.to_json() for dataset in self.datasets]
        }
