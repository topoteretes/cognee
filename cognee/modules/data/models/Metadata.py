from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import UUID, Column, DateTime, String, ForeignKey
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base


class Metadata(Base):
    __tablename__ = "metadata_table"

    id = Column(UUID, primary_key=True, default=uuid4)
    metadata_repr = Column(String)
    metadata_source = Column(String)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    data_id = Column(UUID, ForeignKey("data.id", ondelete="CASCADE"), primary_key=False)
    data = relationship("Data", back_populates="metadata_relationship")
