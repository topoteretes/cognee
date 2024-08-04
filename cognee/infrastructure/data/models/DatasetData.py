from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, UUID, ForeignKey
from cognee.infrastructure.databases.relational import Base

class DatasetData(Base):
    __tablename__ = "dataset_data"

    id = Column(UUID, primary_key=True, default=uuid4)

    dataset_id = Column(UUID, ForeignKey("dataset.id"), nullable=False)
    data_id = Column(UUID, ForeignKey("data.id"), nullable=False)
