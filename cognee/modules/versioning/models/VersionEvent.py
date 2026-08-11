from uuid import uuid4
from datetime import datetime, timezone, timedelta
from sqlalchemy import UUID, Column, DateTime, Integer, String, Index

from cognee.infrastructure.databases.relational import Base

DEFAULT_RETENTION_DAYS = 30


class VersionEvent(Base):
    """Append-only log of every mutating operation on a dataset.

    Each row records one user-triggered operation (ADD / COGNIFY / FORGET)
    along with a JSON payload.  For ADD events the payload contains the
    serialized DataPoint JSON (captured in add_data_points so the full node
    data is available).  For FORGET events the payload contains the node
    slugs and edge names that were hard-deleted, so undo_forget can locate
    the matching ADD events and signal re-ingestion.

    Sequencing
    ----------
    ``sequence_number`` is a per-dataset monotonically increasing counter so
    callers can reconstruct the exact operation order for time-travel.

    Retention window
    ----------------
    ``expires_at`` defaults to ``created_at + 30 days``.  Hard-delete is only
    allowed after ``expires_at``; before that, undo_forget can reverse a FORGET
    event.
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

    # Monotonically increasing sequence per dataset for time-travel ordering.
    sequence_number = Column(Integer, nullable=False, default=1)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    # After this timestamp, the inverse_payload may be compacted / hard-deleted
    # is safe.  Defaults to created_at + DEFAULT_RETENTION_DAYS.
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Set when this event is reversed via undo_forget etc.
    undone_at = Column(DateTime(timezone=True), nullable=True)

    # JSON string storing the operation payload.
    # ADD events:    {"node_ids": ["uuid", ...], "datapoints": [DataPoint.to_json(), ...]}
    # COGNIFY events: {"node_ids": [...], "edge_ids": [...]}
    # FORGET events: {"node_slugs": ["uuid", ...], "edge_slugs": ["rel_name", ...]}
    # Stored as plain text to stay compatible with SQLite (no JSON column type).
    payload = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_version_events_dataset_id", "dataset_id"),
        Index("idx_version_events_data_id", "data_id"),
        Index("idx_version_events_operation", "operation"),
        Index("idx_version_events_created_at", "created_at"),
        Index("idx_version_events_sequence", "dataset_id", "sequence_number"),
    )
