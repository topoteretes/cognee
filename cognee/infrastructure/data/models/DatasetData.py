from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, UUID, ForeignKey, PrimaryKeyConstraint, UniqueConstraint
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base

class DatasetData(Base):
    __tablename__ = "dataset_data"

    id = Column(UUID, primary_key=True, default=uuid4)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    dataset_id = Column(UUID, ForeignKey("dataset.id"), nullable=False)
    data_id = Column(UUID, ForeignKey("data.id"), nullable=False)
    __table_args__ = (
        UniqueConstraint('dataset_id', 'data_id', name='uix_dataset_data'),
    )

    acls = relationship('ACL', back_populates='document')
