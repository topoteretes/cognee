from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, UUID
from cognee.infrastructure.databases.relational import Base


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID, primary_key=True, default=uuid4)

    text = Column(String)
    query_type = Column(String)
    user_id = Column(UUID)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
