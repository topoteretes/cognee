"""Expand dataset database with json connection field

Revision ID: 46a6ce2bd2b2
Revises: 76625596c5c3
Create Date: 2025-11-25 17:56:28.938931

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "46a6ce2bd2b2"
down_revision: Union[str, None] = "76625596c5c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    vector_database_connection_info_column = _get_column(
        insp, "dataset_database", "vector_database_connection_info"
    )
    if not vector_database_connection_info_column:
        op.add_column(
            "dataset_database",
            sa.Column(
                "vector_database_connection_info",
                sa.JSON(),
                unique=False,
                nullable=False,
                default={},
            ),
        )

    graph_database_connection_info_column = _get_column(
        insp, "dataset_database", "graph_database_connection_info"
    )
    if not graph_database_connection_info_column:
        op.add_column(
            "dataset_database",
            sa.Column(
                "graph_database_connection_info",
                sa.JSON(),
                unique=False,
                nullable=False,
                default={},
            ),
        )


def downgrade() -> None:
    op.drop_column("dataset_database", "vector_database_connection_info")
    op.drop_column("dataset_database", "graph_database_connection_info")
