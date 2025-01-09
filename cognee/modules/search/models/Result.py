from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import Column, DateTime, Text, UUID
from cognee.infrastructure.databases.relational import Base


class Result(Base):
    __tablename__ = "results"

    id = Column(UUID, primary_key=True, default=uuid4)

    value = Column(Text)
    query_id = Column(UUID)
    user_id = Column(UUID, index=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
