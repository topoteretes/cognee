from uuid import uuid4, UUID
from enum import Enum
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Mapped
from sqlalchemy import Column, Text, DateTime, UUID as SQLAlchemy_UUID, Integer, Enum as SQLEnum
from cognee.infrastructure.databases.relational import Base


class SyncStatus(str, Enum):
    """Enumeration of possible sync operation statuses."""

    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncOperation(Base):
    """
    Database model for tracking sync operations.

    This model stores information about background sync operations,
    allowing users to monitor progress and query the status of their sync requests.
    """

    __tablename__ = "sync_operations"

    # Primary identifiers
    id = Column(SQLAlchemy_UUID, primary_key=True, default=uuid4, doc="Database primary key")
    run_id = Column(Text, unique=True, index=True, doc="Public run ID returned to users")

    # Status and progress tracking
    status = Column(
        SQLEnum(SyncStatus), default=SyncStatus.STARTED, doc="Current status of the sync operation"
    )
    progress_percentage = Column(Integer, default=0, doc="Progress percentage (0-100)")

    # Operation metadata
    dataset_id = Column(SQLAlchemy_UUID, index=True, doc="ID of the dataset being synced")
    dataset_name = Column(Text, doc="Name of the dataset being synced")
    user_id = Column(SQLAlchemy_UUID, index=True, doc="ID of the user who initiated the sync")

    # Timing information
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        doc="When the sync was initiated",
    )
    started_at = Column(DateTime(timezone=True), doc="When the actual sync processing began")
    completed_at = Column(
        DateTime(timezone=True), doc="When the sync finished (success or failure)"
    )

    # Operation details
    total_records = Column(Integer, doc="Total number of records to sync")
    processed_records = Column(Integer, default=0, doc="Number of records successfully processed")
    bytes_transferred = Column(Integer, default=0, doc="Total bytes transferred to cloud")

    # Error handling
    error_message = Column(Text, doc="Error message if sync failed")
    retry_count = Column(Integer, default=0, doc="Number of retry attempts")

    # Additional metadata (can be added later when needed)
    # cloud_endpoint = Column(Text, doc="Cloud endpoint used for sync")
    # compression_enabled = Column(Text, doc="Whether compression was used")

    def to_json(self) -> dict:
        """Convert sync operation to JSON representation."""
        return {
            "id": str(self.id),
            "run_id": self.run_id,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "dataset_id": str(self.dataset_id),
            "dataset_name": self.dataset_name,
            "user_id": str(self.user_id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_records": self.total_records,
            "processed_records": self.processed_records,
            "bytes_transferred": self.bytes_transferred,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }

    def get_duration_seconds(self) -> Optional[float]:
        """Get the duration of the sync operation in seconds."""
        if not self.created_at:
            return None

        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.created_at).total_seconds()

    def get_progress_info(self) -> dict:
        """Get comprehensive progress information."""
        return {
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "records_processed": f"{self.processed_records or 0}/{self.total_records or 'unknown'}",
            "bytes_transferred": self.bytes_transferred or 0,
            "duration_seconds": self.get_duration_seconds(),
            "error_message": self.error_message,
        }
