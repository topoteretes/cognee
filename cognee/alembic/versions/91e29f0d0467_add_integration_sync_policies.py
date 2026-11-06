"""Add integration_sync_policies: when a principal's connections sync (SDK-1434)

Revision ID: 91e29f0d0467
Revises: dd760c09e075
Create Date: 2026-11-05 15:30:00.000000

One row per owner and connection says how often a connection syncs, on which
weekdays, in which timezone, and whether it is paused. ``credential_id`` NULL is
the owner's default for every connection it owns. The owner is a principal
(``principals.id``, a User or a Tenant), the same polymorphism permissions use,
and no column names a provider or a dataset: a new connector is a new row,
never a new column.

A table, not a key in ``integration_credentials.provider_metadata`` or in
``principal_configuration``: the due scan asks for every row whose
``next_due_at`` has passed, which over JSON is a query no index serves.

Nothing is backfilled. A connection without a row keeps exactly the behaviour
it has today.

Idempotent: the table and each index are inspector-guarded on their own, so an
upgrade interrupted between two statements finishes on rerun. ``downgrade``
drops the table and loses the saved schedules; nothing else reads them.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "91e29f0d0467"
down_revision: str | None = "dd760c09e075"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "integration_sync_policies"

# (name, columns, unique, partial condition)
INDEXES = (
    (
        "uq_integration_sync_policies_connection",
        ["principal_id", "credential_id"],
        True,
        "credential_id IS NOT NULL",
    ),
    ("uq_integration_sync_policies_default", ["principal_id"], True, "credential_id IS NULL"),
    ("ix_integration_sync_policies_due", ["next_due_at"], False, "NOT paused"),
    ("ix_integration_sync_policies_credential_id", ["credential_id"], False, None),
)


def _inspector():
    """A fresh inspector per check: one created before a DDL statement does not see it."""
    return sa.inspect(op.get_bind())


def _has_index(name: str) -> bool:
    return any(index["name"] == name for index in _inspector().get_indexes(TABLE_NAME))


def upgrade() -> None:
    if not _inspector().has_table(TABLE_NAME):
        op.create_table(
            TABLE_NAME,
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column(
                "principal_id",
                sa.Uuid(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "credential_id",
                sa.Uuid(),
                sa.ForeignKey("integration_credentials.id", ondelete="CASCADE"),
                nullable=True,
            ),
            sa.Column("interval_minutes", sa.Integer(), nullable=False),
            sa.Column("weekdays", sa.JSON(), nullable=False),
            sa.Column("timezone", sa.String(), nullable=False),
            sa.Column("paused", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column(
                "paused_by",
                sa.Uuid(),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("paused_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("next_due_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )

    for name, columns, unique, condition in INDEXES:
        if _has_index(name):
            continue
        where = {}
        if condition is not None:
            where = {
                "postgresql_where": sa.text(condition),
                "sqlite_where": sa.text(condition),
            }
        op.create_index(name, TABLE_NAME, columns, unique=unique, **where)


def downgrade() -> None:
    if _inspector().has_table(TABLE_NAME):
        op.drop_table(TABLE_NAME)
