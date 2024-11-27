from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import UUID, Column, DateTime, String, ForeignKey, relationship

from cognee.infrastructure.databases.relational import Base


class Metadata(Base):
    __tablename__ = "metadata_table"

    id = Column(UUID, primary_key=True, default=uuid4)
    metadata_repr = Column(String)
    metadata_source = Column(String)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc)
    )

    dataset_id = Column(UUID, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key = True)
    data_id = Column(UUID, ForeignKey("data.id", ondelete="CASCADE"), primary_key = True)