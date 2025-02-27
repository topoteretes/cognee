from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class RolePermission(Base):
    __tablename__ = "role_permissions"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    role_id = Column(UUID, ForeignKey("roles.id"), primary_key=True)
    permission_id = Column(UUID, ForeignKey("permissions.id"), primary_key=True)
