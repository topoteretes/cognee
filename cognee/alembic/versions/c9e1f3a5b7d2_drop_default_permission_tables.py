"""Drop the unread default-permission tables

Revision ID: c9e1f3a5b7d2
Revises: b8d0f2a4c6e8
Create Date: 2026-10-02 00:00:00.000000

tenant_default_permissions, role_default_permissions and
user_default_permissions were written by the give_default_permission_to_*
methods and read by nothing. The capability layer (principal_capabilities,
f6b8d0a2c4e6) is the tenant-scoped grant model; these tables only looked like
one. The models and the write methods go with them. Downgrade recreates the
tables empty: their rows never had an effect, so there is nothing to restore.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c9e1f3a5b7d2"
down_revision: str | None = "b8d0f2a4c6e8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (table, principal column, principal foreign key)
TABLES = (
    ("tenant_default_permissions", "tenant_id", "tenants.id"),
    ("role_default_permissions", "role_id", "roles.id"),
    ("user_default_permissions", "user_id", "users.id"),
)


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())
    for table_name, _, _ in TABLES:
        if table_name in existing:
            op.drop_table(table_name)
        else:
            print(f"{table_name} table doesn't exist, skipping drop")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing = set(inspector.get_table_names())
    for table_name, principal_column, principal_target in TABLES:
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
