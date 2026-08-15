from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, Uuid
from cognee.infrastructure.databases.relational import Base


class UserRole(Base):
    __tablename__ = "user_roles"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(Uuid, ForeignKey("users.id"), primary_key=True)
    role_id = Column(Uuid, ForeignKey("roles.id"), primary_key=True)
