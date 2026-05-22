"""change node label and type to text

Revision ID: 760ef4f08ef0
Revises: 24f5d4f64d0d
Create Date: 2026-05-21 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "760ef4f08ef0"
down_revision: Union[str, None] = "24f5d4f64d0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name):
    for col in inspector.get_columns(table):
        if col["name"] == name:
            return col
    return None


def _is_varchar_with_length(col):
    if col is None:
        return False
    col_type = col["type"]
    return (
        isinstance(col_type, sa.String)
        and not isinstance(col_type, sa.Text)
        and getattr(col_type, "length", None)
    )


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "nodes" not in insp.get_table_names():
        return

    if conn.dialect.name == "sqlite":
        return

    label_col = _get_column(insp, "nodes", "label")
    if _is_varchar_with_length(label_col):
        op.alter_column(
            "nodes",
            "label",
            existing_type=sa.String(label_col["type"].length),
            type_=sa.Text(),
            existing_nullable=True,
        )

    type_col = _get_column(insp, "nodes", "type")
    if _is_varchar_with_length(type_col):
        op.alter_column(
            "nodes",
            "type",
            existing_type=sa.String(type_col["type"].length),
            type_=sa.Text(),
            existing_nullable=False,
        )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "nodes" not in insp.get_table_names():
        return

    if conn.dialect.name == "sqlite":
        return

    op.alter_column(
        "nodes",
        "label",
        existing_type=sa.Text(),
        type_=sa.String(255),
        existing_nullable=True,
    )

    op.alter_column(
        "nodes",
        "type",
        existing_type=sa.Text(),
        type_=sa.String(255),
        existing_nullable=False,
    )
