from uuid import uuid4, UUID as UUID_t
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy import Column, Text, DateTime, UUID
from cognee.infrastructure.databases.relational import Base
from .DatasetData import DatasetData


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    name: Mapped[str] = mapped_column(Text)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    owner_id: Mapped[UUID_t] = mapped_column(UUID(as_uuid=True), index=True)

    acls = relationship("ACL", back_populates="dataset", cascade="all, delete-orphan")

    data: Mapped[List["Data"]] = relationship(
        "Data",
        secondary=DatasetData.__tablename__,
        back_populates="datasets",
        lazy="noload",
        cascade="all, delete",
    )

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat() if self.updated_at else None,
            "ownerId": str(self.owner_id),
            "data": [data.to_json() for data in self.data],
        }
