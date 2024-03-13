from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, UUID, ForeignKey
from cognee.infrastructure.databases.relational import ModelBase

class DatasetData(ModelBase):
    __tablename__ = "dataset_data"

    id = Column(UUID, primary_key = True, default = uuid4())

    created_at = Column(DateTime, default = datetime.now(timezone.utc))

    dataset_id = Column("dataset", UUID, ForeignKey("dataset.id"), primary_key = True)
    data_id = Column("data", UUID, ForeignKey("data.id"), primary_key = True)
