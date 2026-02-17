"""create principal configuration table

Revision ID: 18f01b0a0b4c
Revises: 84e5d08260d6
Create Date: 2026-02-17 12:49:02.909973

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "18f01b0a0b4c"
down_revision: Union[str, None] = "84e5d08260d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    table_name = "principal_configuration"
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_name not in inspector.get_table_names():
        op.create_table(
            table_name,
            sa.Column(
                "owner_id",
                sa.UUID(),
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                primary_key=True,
                index=True,
            ),
            sa.Column("name", sa.String(), unique=False, nullable=False),
            sa.Column("configuration", sa.JSON()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )
    else:
        print(f"{table_name} table already exists, skipping creation")


def downgrade() -> None:
    table_name = "principal_configuration"
    connection = op.get_bind()
    inspector = sa.inspect(connection)

    if table_name not in inspector.get_table_names():
        op.drop_table(table_name)
    else:
        print(f"{table_name} table doesn't exist, skipping downgrade")
