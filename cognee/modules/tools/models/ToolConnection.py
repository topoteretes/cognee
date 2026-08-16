from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, LargeBinary, SmallInteger, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational.ModelBase import Base


class ToolConnection(Base):
    """A user-authorized external database connection for recall tool calls.

    Deliberately a separate table from ``integration_credentials``: these rows
    are named per owner (UNIQUE(user_id, name)), carry no OAuth semantics, and
    have no external account id to route webhooks by. The encrypted-payload
    column set (``ciphertext``/``nonce``/``encryption_version``/``key_id``) is
    the same contract, decrypted through
    :mod:`cognee.modules.integrations.crypto`, so key rotation covers both
    tables at once.

    The connection string (DSN) lives only in ``ciphertext``. ``options``
    holds non-secret settings: ``allowed_tables``, ``max_rows``,
    ``description``.
    """

    __tablename__ = "tool_connections"

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_tool_connections_user_name"),)

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    user_id: Mapped[UUID] = mapped_column(Uuid, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False)

    # "postgres" | "sqlite" in v1; inferred from the DSN at registration.
    provider: Mapped[str] = mapped_column(String, nullable=False)

    # AES-GCM-encrypted {"connection_string": ...} payload — see
    # cognee.modules.integrations.crypto for the keyring/rotation contract.
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    key_id: Mapped[str] = mapped_column(String, nullable=False, default="1")

    options: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
