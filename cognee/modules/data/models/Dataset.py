from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, Text, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(Text)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    owner_id = Column(UUID, index=True)
    tenant_id = Column(UUID, index=True, nullable=True)

    acls = relationship("ACL", back_populates="dataset", cascade="all, delete-orphan")
    configuration = relationship(
        "DatasetConfiguration",
        back_populates="dataset",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Data rows are dataset-scoped (Data.dataset_id); this is a read-only view
    # over that column — membership no longer has its own table. Writes go
    # through Data.dataset_id directly.
    data: Mapped[List["Data"]] = relationship(
        "Data",
        primaryjoin="Dataset.id == foreign(Data.dataset_id)",
        lazy="noload",
        viewonly=True,
    )

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "ownerId": str(self.owner_id),
            "tenantId": str(self.tenant_id),
            "data": [data.to_json() for data in self.data],
        }
