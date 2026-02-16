"""Model for storing refresh token hashes to support token rotation and revocation."""

from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, String, UUID
from cognee.infrastructure.databases.relational import Base


class RefreshToken(Base):
    """Stores hashed refresh tokens for a user. Used for refresh and revoke flows."""

    __tablename__ = "refresh_tokens"

    id = Column(UUID, primary_key=True, default=uuid4)
    user_id = Column(UUID, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
