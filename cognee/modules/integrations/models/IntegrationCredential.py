from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from cognee.infrastructure.databases.relational.ModelBase import Base
from sqlalchemy import JSON, DateTime, Index, LargeBinary, SmallInteger, String
from sqlalchemy import UUID as SAUUID
from sqlalchemy.orm import Mapped, mapped_column


class IntegrationCredential(Base):
    """A single third-party connector connection.

    One table for every connector — Slack, Notion, GitHub, ... — discriminated
    by ``provider``, not one table per connector. Adding connector #N is a new
    row, never a new table.

    ``provider_account_id`` is the external workspace/org id (Slack ``team_id``,
    a Notion workspace id, a GitHub org id). The UNIQUE(provider,
    provider_account_id) constraint is what lets an inbound webhook that
    carries only that external id resolve back to exactly one owning user — so
    a workspace can belong to only one cognee user, and the routing is never
    ambiguous. It is scoped by ``user_id`` (the connecting cognee user), not a
    tenant — this is a single/multi-user SDK concept, not a SaaS tenant.
    """

    __tablename__ = "integration_credentials"

    __table_args__ = (
        Index(
            "ix_integration_credentials_provider_account",
            "provider",
            "provider_account_id",
            unique=True,
        ),
    )

    id: Mapped[UUID] = mapped_column(SAUUID, primary_key=True, default=uuid4)

    user_id: Mapped[UUID] = mapped_column(SAUUID, nullable=False, index=True)

    provider: Mapped[str] = mapped_column(String, nullable=False)
    provider_account_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    account_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # 'oauth2' | 'api_key' — most connectors are OAuth, a few take a raw key.
    auth_type: Mapped[str] = mapped_column(String, nullable=False, default="oauth2")
    scopes: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Non-secret, connector-specific display/routing data (e.g. Slack's
    # bot_user_id, enterprise_id, incoming-webhook URL). Secret material never
    # goes here — only in ciphertext.
    provider_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # AES-GCM-encrypted token payload (access + refresh token, or a raw API
    # key). encryption_version dispatches decryption so a future KMS envelope
    # scheme is a version bump, not a schema migration.
    ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    nonce: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)

    # Which key in the encryption keyring this row was written under. Rows keep
    # decrypting under their original id after a key rotation (new active id);
    # see modules.integrations.crypto.
    key_id: Mapped[str] = mapped_column(String, nullable=False, default="1")

    token_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")

    last_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sync_status: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
