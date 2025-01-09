from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, UUID
from cognee.infrastructure.databases.relational import Base


class DatasetData(Base):
    __tablename__ = "dataset_data"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    dataset_id = Column(UUID, ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True)
    data_id = Column(UUID, ForeignKey("data.id", ondelete="CASCADE"), primary_key=True)
