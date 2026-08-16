from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, Uuid
from cognee.infrastructure.databases.relational import Base


class UserDefaultPermissions(Base):
    __tablename__ = "user_default_permissions"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user_id = Column(Uuid, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    permission_id = Column(
        Uuid,
        ForeignKey(
            "permissions.id", ondelete="CASCADE"
        ),  # cascade deletion when Permission is deleted
        primary_key=True,
    )
