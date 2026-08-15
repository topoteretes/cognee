from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, Uuid
from cognee.infrastructure.databases.relational import Base


class UserTenant(Base):
    __tablename__ = "user_tenants"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(Uuid, ForeignKey("users.id"), primary_key=True)
    tenant_id = Column(Uuid, ForeignKey("tenants.id"), primary_key=True)
