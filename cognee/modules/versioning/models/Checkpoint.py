from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import UUID, Column, DateTime, String, Index

from cognee.infrastructure.databases.relational import Base


class Checkpoint(Base):
    """Materialized snapshot of the alive node/edge set for a dataset.

    Checkpoints store IDs only (node slugs + edge relationship names as
    JSON strings), not full graph/vector data.  They are cheap to create
    and allow time-travel diffing without duplicating the stores.

    A checkpoint is created explicitly via ``create_checkpoint()`` or
    automatically after each COGNIFY event.
    """

    __tablename__ = "versioning_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    user_id = Column(UUID(as_uuid=True), nullable=True)

    # Optional human-readable label, e.g. "after-ingestion-v2"
    label = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # JSON-encoded list of node slugs (UUIDs as strings) that were alive at
    # checkpoint time, e.g. '["slug-a", "slug-b"]'
    node_slugs = Column(String, nullable=False, default="[]")

    # JSON-encoded list of edge relationship names alive at checkpoint time
    edge_slugs = Column(String, nullable=False, default="[]")

    __table_args__ = (
        Index("idx_versioning_checkpoints_dataset_id", "dataset_id"),
        Index("idx_versioning_checkpoints_created_at", "created_at"),
    )
