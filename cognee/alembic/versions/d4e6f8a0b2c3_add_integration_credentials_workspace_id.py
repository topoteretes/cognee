"""add_integration_credentials_workspace_id

Revision ID: d4e6f8a0b2c3
Revises: c3d5e7f9a1b2
Create Date: 2026-08-03 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e6f8a0b2c3"
down_revision: Union[str, None] = "c3d5e7f9a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    columns = {col["name"] for col in insp.get_columns("integration_credentials")}
    if "workspace_id" not in columns:
        op.add_column(
            "integration_credentials",
            sa.Column("workspace_id", sa.UUID(), nullable=True),
        )

    indexes = {idx["name"] for idx in insp.get_indexes("integration_credentials")}
    if "ix_integration_credentials_workspace_id" not in indexes:
        op.create_index(
            "ix_integration_credentials_workspace_id",
            "integration_credentials",
            ["workspace_id"],
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    indexes = {idx["name"] for idx in insp.get_indexes("integration_credentials")}
    if "ix_integration_credentials_workspace_id" in indexes:
        op.drop_index(
            "ix_integration_credentials_workspace_id", table_name="integration_credentials"
        )

    columns = {col["name"] for col in insp.get_columns("integration_credentials")}
    if "workspace_id" in columns:
        op.drop_column("integration_credentials", "workspace_id")
