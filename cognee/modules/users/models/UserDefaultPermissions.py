from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class UserDefaultPermissions(Base):
    __tablename__ = "user_default_permissions"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(UUID, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    permission_id = Column(
        UUID,
        ForeignKey(
            "permissions.id", ondelete="CASCADE"
        ),  # cascade deletion when Permission is deleted
        primary_key=True,
    )
