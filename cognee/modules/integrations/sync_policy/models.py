from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    false,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational.ModelBase import Base


class IntegrationSyncPolicy(Base):
    """When a principal's connections sync, and whether they are paused.

    ``credential_id`` NULL is the principal's default for every connection it
    owns; set, it is that connection's own policy. The owner is a principal
    (a ``User`` or a ``Tenant``), and no column names a provider or a dataset,
    so a new connector is a new row, never a new column.

    ``credential_id`` references the host's own ``integration_credentials`` table
    (the SDK's, or Cloud's). Import this model only where one is defined: the SDK
    registers both through ``cognee.modules.integrations.models``.
    """

    __tablename__ = "integration_sync_policies"

    __table_args__ = (
        # Two partial unique indexes instead of one: two NULL credential ids are
        # not equal to each other, so a plain unique index would allow several
        # default rows for one principal.
        Index(
            "uq_integration_sync_policies_connection",
            "principal_id",
            "credential_id",
            unique=True,
            postgresql_where=text("credential_id IS NOT NULL"),
            sqlite_where=text("credential_id IS NOT NULL"),
        ),
        Index(
            "uq_integration_sync_policies_default",
            "principal_id",
            unique=True,
            postgresql_where=text("credential_id IS NULL"),
            sqlite_where=text("credential_id IS NULL"),
        ),
        Index(
            "ix_integration_sync_policies_due",
            "next_due_at",
            postgresql_where=text("NOT paused"),
            sqlite_where=text("NOT paused"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)

    # A plain index too: the partial ones above do not serve user deletion, nor the cascade
    # from principals, which look rows up by principal alone.
    principal_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("principals.id", ondelete="CASCADE"), nullable=False, index=True
    )
    credential_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("integration_credentials.id", ondelete="CASCADE"), index=True
    )

    # Written only through SyncPolicy, which validates them and owns their defaults.
    interval_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    weekdays: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    timezone: Mapped[str] = mapped_column(String, nullable=False)

    paused: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    paused_by: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Always written in UTC: SQLite drops the offset on read, so any other zone
    # would compare wrong in the due scan and in the claim.
    next_due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
