from datetime import datetime, timezone
from sqlalchemy import Column, ForeignKey, DateTime, UUID
from cognee.infrastructure.databases.relational import Base


class RoleDefaultPermissions(Base):
    __tablename__ = "role_default_permissions"

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    role_id = Column(
        UUID,
        ForeignKey("roles.id", ondelete="CASCADE"),  # cascade deletion when Role is deleted
        primary_key=True,
    )
    permission_id = Column(
        UUID,
        ForeignKey(
            "permissions.id", ondelete="CASCADE"
        ),  # cascade deletion when Permission is deleted
        primary_key=True,
    )
