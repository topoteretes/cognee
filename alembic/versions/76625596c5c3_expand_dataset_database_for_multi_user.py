"""Expand dataset database for multi user

Revision ID: 76625596c5c3
Revises: 211ab850ef3d
Create Date: 2025-10-30 12:55:20.239562

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "76625596c5c3"
down_revision: Union[str, None] = "c946955da633"
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

    vector_database_provider_column = _get_column(
        insp, "dataset_database", "vector_database_provider"
    )
    if not vector_database_provider_column:
        op.add_column(
            "dataset_database",
            sa.Column(
                "vector_database_provider",
                sa.String(),
                unique=False,
                nullable=False,
                server_default="lancedb",
            ),
        )

    graph_database_provider_column = _get_column(
        insp, "dataset_database", "graph_database_provider"
    )
    if not graph_database_provider_column:
        op.add_column(
            "dataset_database",
            sa.Column(
                "graph_database_provider",
                sa.String(),
                unique=False,
                nullable=False,
                server_default="kuzu",
            ),
        )

    vector_database_url_column = _get_column(insp, "dataset_database", "vector_database_url")
    if not vector_database_url_column:
        op.add_column(
            "dataset_database",
            sa.Column("vector_database_url", sa.String(), unique=False, nullable=True),
        )

    graph_database_url_column = _get_column(insp, "dataset_database", "graph_database_url")
    if not graph_database_url_column:
        op.add_column(
            "dataset_database",
            sa.Column("graph_database_url", sa.String(), unique=False, nullable=True),
        )

    vector_database_key_column = _get_column(insp, "dataset_database", "vector_database_key")
    if not vector_database_key_column:
        op.add_column(
            "dataset_database",
            sa.Column("vector_database_key", sa.String(), unique=False, nullable=True),
        )

    graph_database_key_column = _get_column(insp, "dataset_database", "graph_database_key")
    if not graph_database_key_column:
        op.add_column(
            "dataset_database",
            sa.Column("graph_database_key", sa.String(), unique=False, nullable=True),
        )


def downgrade() -> None:
    op.drop_column("dataset_database", "vector_database_provider")
    op.drop_column("dataset_database", "graph_database_provider")
    op.drop_column("dataset_database", "vector_database_url")
    op.drop_column("dataset_database", "graph_database_url")
    op.drop_column("dataset_database", "vector_database_key")
    op.drop_column("dataset_database", "graph_database_key")
