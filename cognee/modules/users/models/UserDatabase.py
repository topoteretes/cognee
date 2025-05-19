from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String, UUID, ForeignKey
from cognee.infrastructure.databases.relational import Base


class UserDatabase(Base):
    __tablename__ = "user_database"

    owner_id = Column(
        UUID, ForeignKey("principals.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    dataset_id = Column(UUID, ForeignKey("datasets.id", ondelete="CASCADE"))

    vector_database_loc = Column(String, unique=True, nullable=False)
    graph_database_loc = Column(String, unique=True, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
