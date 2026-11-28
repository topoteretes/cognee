from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational.ModelBase import Base


class UserRefreshToken(Base):
    """One browser login that can be renewed without the password.

    The token itself lives only in the browser's refresh cookie; the row holds
    its hash. See ``cognee.modules.users.authentication.refresh_token``.
    """

    __tablename__ = "user_refresh_token"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("principals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(nullable=False, unique=True, index=True)
    # Fingerprint of the password the token was issued under, keyed with the
    # token secret: a change of either ends the login.
    password_fingerprint: Mapped[str] = mapped_column(nullable=False)
    # When the login started: it ends a maximum lifetime later, however often it is used.
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    # Moved forward on every use, so it is the end of the allowed idle time.
    # Indexed for the removal of expired rows on login.
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
