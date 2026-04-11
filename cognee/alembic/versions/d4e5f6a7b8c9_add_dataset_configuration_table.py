"""add dataset_configuration table

Revision ID: d4e5f6a7b8c9
Revises:
Create Date: 2026-04-04
"""

from alembic import op
import sqlalchemy as sa

revision = "d4e5f6a7b8c9"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "dataset_configurations" not in insp.get_table_names():
        op.create_table(
            "dataset_configurations",
            sa.Column("id", sa.UUID(), primary_key=True),
            sa.Column(
                "dataset_id",
                sa.UUID(),
                sa.ForeignKey("datasets.id", ondelete="CASCADE"),
                unique=True,
                nullable=False,
            ),
            sa.Column("graph_schema", sa.JSON(), nullable=True),
            sa.Column("custom_prompt", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True)),
            sa.Column("updated_at", sa.DateTime(timezone=True)),
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "dataset_configurations" in insp.get_table_names():
        op.drop_table("dataset_configurations")
