"""Add granted_by to principal_capabilities

Revision ID: b8d0f2a4c6e8
Revises: f6b8d0a2c4e6
Create Date: 2026-09-23 00:00:00.000000

Records which user made each capability grant, so it is possible to trace how
a principal came to hold a capability. Nullable with SET NULL: grants written
through the SDK have no requester, and a grant outlives the user who made it.
No backfill: rows written before this column existed have no known granter.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8d0f2a4c6e8"
down_revision: str | None = "f6b8d0a2c4e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_NAME = "principal_capabilities"
COLUMN_NAME = "granted_by"
FK_NAME = "fk_principal_capabilities_granted_by_users"


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_foreign_key(inspector: sa.Inspector, table_name: str, fk_name: str) -> bool:
    return any(
        foreign_key.get("name") == fk_name for foreign_key in inspector.get_foreign_keys(table_name)
    )


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if TABLE_NAME not in inspector.get_table_names():
        return

    if not _has_column(inspector, TABLE_NAME, COLUMN_NAME):
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table(TABLE_NAME) as batch_op:
                batch_op.add_column(sa.Column(COLUMN_NAME, sa.UUID(), nullable=True))
        else:
            op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.UUID(), nullable=True))

    if op.get_context().dialect.name != "sqlite":
        inspector = sa.inspect(connection)
        if not _has_foreign_key(inspector, TABLE_NAME, FK_NAME):
            op.create_foreign_key(
                FK_NAME,
                TABLE_NAME,
                "users",
                [COLUMN_NAME],
                ["id"],
                ondelete="SET NULL",
            )


def downgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if TABLE_NAME not in inspector.get_table_names() or not _has_column(
        inspector, TABLE_NAME, COLUMN_NAME
    ):
        return

    if op.get_context().dialect.name != "sqlite" and _has_foreign_key(
        inspector, TABLE_NAME, FK_NAME
    ):
        op.drop_constraint(FK_NAME, TABLE_NAME, type_="foreignkey")

    if op.get_context().dialect.name == "sqlite":
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.drop_column(COLUMN_NAME)
    else:
        op.drop_column(TABLE_NAME, COLUMN_NAME)
