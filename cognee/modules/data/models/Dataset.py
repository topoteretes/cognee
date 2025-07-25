from uuid import uuid4
from typing import List
from datetime import datetime, timezone
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy import Column, Text, DateTime, UUID
from cognee.infrastructure.databases.relational import Base
from .DatasetData import DatasetData


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID, primary_key=True, default=uuid4)

    name = Column(Text)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    owner_id = Column(UUID, index=True)

    acls = relationship("ACL", back_populates="dataset", cascade="all, delete-orphan")

    data: Mapped[List["Data"]] = relationship(
        "Data",
        secondary=DatasetData.__tablename__,
        back_populates="datasets",
        lazy="noload",
        cascade="all, delete",
    )
