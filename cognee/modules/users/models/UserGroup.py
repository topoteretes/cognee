from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class UserGroup(Base):
    __tablename__ = "user_groups"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(UUID, ForeignKey("users.id"), primary_key=True)
    group_id = Column(UUID, ForeignKey("groups.id"), primary_key=True)
