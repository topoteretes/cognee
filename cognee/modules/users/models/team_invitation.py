"""Single-use, email-bound invitations; only token hashes are stored."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import UUID, Column, DateTime, ForeignKey, String

from cognee.infrastructure.databases.relational import Base


class TeamInvitation(Base):
    __tablename__ = "team_invitations"

    id = Column(UUID, primary_key=True, default=uuid4)
    tenant_id = Column(UUID, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    created_by = Column(UUID, ForeignKey("users.id"), nullable=False)
    email = Column(String, nullable=False)
    token_hash = Column(String(64), unique=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime(timezone=True), nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
