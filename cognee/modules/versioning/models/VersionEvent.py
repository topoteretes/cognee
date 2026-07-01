from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import UUID, Column, DateTime, String, Index

from cognee.infrastructure.databases.relational import Base


class VersionEvent(Base):
    """Append-only log of every mutating operation on a dataset.

    Each row records one user-triggered operation (ADD / COGNIFY / FORGET)
    along with a JSON payload capturing the node slugs and edge names
    affected.  For FORGET events the payload is used by undo_forget to
    clear the ledger soft-delete so the data can be re-ingested.

    The table is append-only by convention: rows are never deleted, only
    updated via ``undone_at`` to signal reversal.
    """

    __tablename__ = "version_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    # Operation type: "ADD" | "COGNIFY" | "FORGET"
    operation = Column(String, nullable=False)

    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    # data_id is null for dataset-level operations (e.g. full dataset forget)
    data_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    # pipeline run that triggered this operation (from Node.pipeline_run_id)
    run_id = Column(UUID(as_uuid=True), nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # Set when this event is reversed via undo_forget / undo_add etc.
    undone_at = Column(DateTime(timezone=True), nullable=True)

    # JSON string: {"node_slugs": ["..."], "edge_slugs": ["..."]}
    # Stored as plain text to stay compatible with SQLite (no JSON column type).
    payload = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_version_events_dataset_id", "dataset_id"),
        Index("idx_version_events_data_id", "data_id"),
        Index("idx_version_events_operation", "operation"),
        Index("idx_version_events_created_at", "created_at"),
    )
