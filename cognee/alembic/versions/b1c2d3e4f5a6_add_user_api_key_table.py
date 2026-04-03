"""add_user_api_key_table

Revision ID: b1c2d3e4f5a6
Revises: f1a2b3c4d5e6
Create Date: 2026-04-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5a6"
down_revision: Union[str, None] = "18f01b0a0b4c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "user_api_key" not in insp.get_table_names():
        op.create_table(
            "user_api_key",
            sa.Column("id", sa.UUID, primary_key=True),
            sa.Column(
                "user_id",
                sa.UUID,
                sa.ForeignKey("principals.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("api_key", sa.String, nullable=False),
            sa.Column("label", sa.String, nullable=True),
            sa.Column("name", sa.String, nullable=True),
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "user_api_key" in insp.get_table_names():
        op.drop_table("user_api_key")
