"""Add the capability layer

Revision ID: f6b8d0a2c4e6
Revises: e7f9a1c3d5b8
Create Date: 2026-08-01 00:00:00.000000

One revision for the whole capability layer, since none of it has reached a
production database: create ``principal_capabilities`` — one row per
(principal, tenant, capability), with ``granted_by`` recording who made the
grant — and drop the three ``*_default_permissions`` tables it replaces, which
were written by nothing that was ever read.

Foreign keys: ``principal_id`` and ``tenant_id`` cascade, so a grant cannot
outlive its holder or its scope; ``granted_by`` is SET NULL, so removing the
granter keeps what they granted. Declared on every dialect at table-creation
time; SQLite stores them but does not enforce them (cognee leaves the
``foreign_keys`` pragma off), which is why the ORM also cascades a user's own
rows (``User.capabilities``). ``tenant_id`` gets its own index because
resolution asks what a set of principals holds in one tenant, which the
primary key (leading on principal_id) does not serve.

Downgrade restores the pre-layer schema exactly: the table goes, the three
default-permission tables come back empty — their rows never had an effect.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6b8d0a2c4e6"
down_revision: str | None = "e7f9a1c3d5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "principal_capabilities"
INDEX_NAME = "ix_principal_capabilities_tenant_id"

# (table, principal column, principal foreign key) of the tables this replaces
DROPPED_TABLES = (
    ("tenant_default_permissions", "tenant_id", "tenants.id"),
    ("role_default_permissions", "role_id", "roles.id"),
    ("user_default_permissions", "user_id", "users.id"),
)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())

    # Existence-guarded like every revision in the chain, so it no-ops on a
    # database that already has the table.
    if TABLE_NAME not in existing:
        op.create_table(
            TABLE_NAME,
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("principal_id", sa.UUID(), nullable=False),
            sa.Column("tenant_id", sa.UUID(), nullable=False),
            sa.Column("capability", sa.String(), nullable=False),
            sa.Column("granted_by", sa.UUID(), nullable=True),
            sa.ForeignKeyConstraint(["principal_id"], ["principals.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["granted_by"],
                ["users.id"],
                name="fk_principal_capabilities_granted_by_users",
                ondelete="SET NULL",
            ),
            sa.PrimaryKeyConstraint("principal_id", "tenant_id", "capability"),
        )
        op.create_index(INDEX_NAME, TABLE_NAME, ["tenant_id"])

    for table_name, _, _ in DROPPED_TABLES:
        if table_name in existing:
            op.drop_table(table_name)
        else:
            print(f"{table_name} table doesn't exist, skipping drop")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())

    if TABLE_NAME in existing:
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
        op.drop_table(TABLE_NAME)

    for table_name, principal_column, principal_target in DROPPED_TABLES:
        if table_name in existing:
            print(f"{table_name} table already exists, skipping downgrade")
            continue
        op.create_table(
            table_name,
            sa.Column("created_at", sa.DateTime(timezone=True)),
            sa.Column(
                principal_column,
                sa.UUID(),
                sa.ForeignKey(principal_target, ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "permission_id",
                sa.UUID(),
                sa.ForeignKey("permissions.id", ondelete="CASCADE"),
                primary_key=True,
            ),
        )
