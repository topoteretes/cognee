from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import UUID, Column, DateTime, Index, String, JSON, Integer, Float
from sqlalchemy.ext.mutable import MutableDict

from cognee.infrastructure.databases.relational import Base


class Data(Base):
    __tablename__ = "data"
    # Dedup is a lookup, not an identity: adding content already present in a
    # dataset (by the same owner/tenant) reuses that row via this index; the id
    # itself carries no content information and stays stable when the
    # document's content changes.
    __table_args__ = (
        Index("data_dataset_content_lookup", "dataset_id", "owner_id", "content_hash"),
    )

    id = Column(UUID, primary_key=True, default=uuid4)
    label = Column(String, nullable=True)
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
    # Dataset that owns this content row. Rows are dataset-scoped: the same
    # content in two datasets is two rows with two ids, so updating one
    # document can never touch another dataset's data. NULL marks a legacy
    # shared row (pre-refactor deterministic id, possibly member of several
    # datasets); those are resolved by membership and split copy-on-write on
    # their first content update.
    dataset_id = Column(UUID, index=True, nullable=True)
    # The pre-refactor data_id this row's identity descends from (flattened —
    # always the original user-visible id, never an intermediate). Set on
    # backfill-split rows and carried forward by the update path so every id
    # ever issued keeps resolving; NULL for rows whose id never changed.
    legacy_id = Column(UUID, index=True, nullable=True)
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
    last_accessed = Column(DateTime(timezone=True), nullable=True)
    importance_weight = Column(Float, nullable=True)

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "label": self.label,
            "extension": self.extension,
            "mimeType": self.mime_type,
            "rawDataLocation": self.raw_data_location,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "nodeSet": self.node_set,
            # "datasets": [dataset.to_json() for dataset in self.datasets]
        }
