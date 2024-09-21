from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey
from cognee.infrastructure.databases.relational import Base, UUID

class DatasetData(Base):
    __tablename__ = "dataset_data"

    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))

    dataset_id = Column(UUID, ForeignKey("datasets.id"), primary_key = True)
    data_id = Column(UUID, ForeignKey("data.id"), primary_key = True)
