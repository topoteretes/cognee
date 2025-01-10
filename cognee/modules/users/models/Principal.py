from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class Principal(Base):
    __tablename__ = "principals"

    id = Column(UUID, primary_key=True, index=True, default=uuid4)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    type = Column(String, nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "principal",
        "polymorphic_on": "type",
    }
