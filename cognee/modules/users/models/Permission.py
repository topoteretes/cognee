from uuid import uuid4
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, Uuid
from cognee.infrastructure.databases.relational import Base


class Permission(Base):
    __tablename__ = "permissions"

    id = Column(Uuid, primary_key=True, index=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    name = Column(String, unique=True, nullable=False, index=True)
