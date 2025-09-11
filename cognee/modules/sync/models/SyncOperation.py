from uuid import uuid4
from enum import Enum
from typing import Optional, List
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Text,
    DateTime,
    UUID as SQLAlchemy_UUID,
    Integer,
    Enum as SQLEnum,
    JSON,
)

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
    dataset_ids = Column(JSON, doc="Array of dataset IDs being synced")
    dataset_names = Column(JSON, doc="Array of dataset names being synced")
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
    total_records_to_sync = Column(Integer, doc="Total number of records to sync")
    total_records_to_download = Column(Integer, doc="Total number of records to download")
    total_records_to_upload = Column(Integer, doc="Total number of records to upload")

    records_downloaded = Column(Integer, default=0, doc="Number of records successfully downloaded")
    records_uploaded = Column(Integer, default=0, doc="Number of records successfully uploaded")
    bytes_downloaded = Column(Integer, default=0, doc="Total bytes downloaded from cloud")
    bytes_uploaded = Column(Integer, default=0, doc="Total bytes uploaded to cloud")

    # Data lineage tracking per dataset
    dataset_sync_hashes = Column(
        JSON, doc="Mapping of dataset_id -> {uploaded: [hashes], downloaded: [hashes]}"
    )

    # Error handling
    error_message = Column(Text, doc="Error message if sync failed")
    retry_count = Column(Integer, default=0, doc="Number of retry attempts")

    def get_duration_seconds(self) -> Optional[float]:
        """Get the duration of the sync operation in seconds."""
        if not self.created_at:
            return None

        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.created_at).total_seconds()

    def get_progress_info(self) -> dict:
        """Get comprehensive progress information."""
        total_records_processed = (self.records_downloaded or 0) + (self.records_uploaded or 0)
        total_bytes_transferred = (self.bytes_downloaded or 0) + (self.bytes_uploaded or 0)

        return {
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "records_processed": f"{total_records_processed}/{self.total_records_to_sync or 'unknown'}",
            "records_downloaded": self.records_downloaded or 0,
            "records_uploaded": self.records_uploaded or 0,
            "bytes_transferred": total_bytes_transferred,
            "bytes_downloaded": self.bytes_downloaded or 0,
            "bytes_uploaded": self.bytes_uploaded or 0,
            "duration_seconds": self.get_duration_seconds(),
            "error_message": self.error_message,
            "dataset_sync_hashes": self.dataset_sync_hashes or {},
        }

    def _get_all_sync_hashes(self) -> List[str]:
        """Get all content hashes for data created/modified during this sync operation."""
        all_hashes = set()
        dataset_hashes = self.dataset_sync_hashes or {}

        for dataset_id, operations in dataset_hashes.items():
            if isinstance(operations, dict):
                all_hashes.update(operations.get("uploaded", []))
                all_hashes.update(operations.get("downloaded", []))

        return list(all_hashes)

    def _get_dataset_sync_hashes(self, dataset_id: str) -> dict:
        """Get uploaded/downloaded hashes for a specific dataset."""
        dataset_hashes = self.dataset_sync_hashes or {}
        return dataset_hashes.get(dataset_id, {"uploaded": [], "downloaded": []})

    def was_data_synced(self, content_hash: str, dataset_id: str = None) -> bool:
        """
        Check if a specific piece of data was part of this sync operation.

        Args:
            content_hash: The content hash to check for
            dataset_id: Optional - check only within this dataset
        """
        if dataset_id:
            dataset_hashes = self.get_dataset_sync_hashes(dataset_id)
            return content_hash in dataset_hashes.get(
                "uploaded", []
            ) or content_hash in dataset_hashes.get("downloaded", [])

        all_hashes = self.get_all_sync_hashes()
        return content_hash in all_hashes
