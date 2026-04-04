"""add dataset_configuration table

Revision ID: a1b2c3d4e5f6
Revises:
Create Date: 2026-04-03
"""

from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
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
    op.drop_table("dataset_configurations")
