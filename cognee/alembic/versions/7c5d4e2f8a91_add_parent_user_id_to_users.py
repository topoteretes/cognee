"""Add parent_user_id to users

Revision ID: 7c5d4e2f8a91
Revises: b2c3d4e5f6a7
Create Date: 2026-04-23 21:35:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7c5d4e2f8a91"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "users"
COLUMN_NAME = "parent_user_id"
FK_NAME = "fk_users_parent_user_id_users"


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
                TABLE_NAME,
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
