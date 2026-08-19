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

    # Mirrors ``Query.dataset_id`` for the query this result answers, so
    # history can be filtered by dataset without joining back to queries.
    dataset_id = Column(UUID, index=True, nullable=True)

    created_at = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
