from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, UUID
from cognee.infrastructure.databases.relational import Base
from .ACLResources import ACLResources


class Resource(Base):
    __tablename__ = "resources"

    id = Column(UUID, primary_key=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    resource_id = Column(UUID, nullable=False)

    acls = relationship("ACL", secondary=ACLResources.__tablename__, back_populates="resources")
