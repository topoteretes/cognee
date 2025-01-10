from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class GroupPermission(Base):
    __tablename__ = "group_permissions"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    group_id = Column(UUID, ForeignKey("groups.id"), primary_key=True)
    permission_id = Column(UUID, ForeignKey("permissions.id"), primary_key=True)
