"""add principal_capabilities table

Revision ID: f6b8d0a2c4e6
Revises: d4e6f8a0b2c3
Create Date: 2026-08-03 00:00:00.000000

Capabilities (tenant-scoped actions such as "manage_users") get a table of
their own instead of riding the ``*DefaultPermissions`` tables. Those reference
the dataset-permission catalog, so capability names would sit next to
read/write/delete/share and be told apart by string filtering, and the
user-level table has no tenant column, so a per-person grant would apply in
every tenant the person belongs to.

``tenant_id`` is part of the key on every row, including rows whose principal
is a role or the tenant itself, where it is derivable: resolution filters on it
directly, and a grant leaking across tenants becomes unrepresentable.

No data migration: nothing wrote capability rows before this table existed.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6b8d0a2c4e6"
down_revision: Union[str, None] = "d4e6f8a0b2c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "principal_capabilities",
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("principal_id", sa.UUID(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("capability", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["principal_id"], ["principals.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("principal_id", "tenant_id", "capability"),
    )
    # Resolution asks "what does this set of principals hold in this tenant",
    # which the primary key (leading on principal_id) does not serve.
    op.create_index(
        "ix_principal_capabilities_tenant_id",
        "principal_capabilities",
        ["tenant_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_principal_capabilities_tenant_id", table_name="principal_capabilities")
    op.drop_table("principal_capabilities")
