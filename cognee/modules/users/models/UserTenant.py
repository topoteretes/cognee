from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class UserTenant(Base):
    __tablename__ = "user_tenants"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(UUID, ForeignKey("users.id"), primary_key=True)
    tenant_id = Column(UUID, ForeignKey("tenants.id"), primary_key=True)
