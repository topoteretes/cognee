from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class UserRole(Base):
    __tablename__ = "user_roles"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(UUID, ForeignKey("users.id"), primary_key=True)
    role_id = Column(UUID, ForeignKey("roles.id"), primary_key=True)
