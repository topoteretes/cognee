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
down_revision: Union[str, None] = "211ab850ef3d"
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

    data = sa.table(
        "dataset_database",
        sa.Column("dataset_id", sa.UUID, primary_key=True, index=True),  # Critical for SQLite
        sa.Column("owner_id", sa.UUID, index=True),
        sa.Column("vector_database_name", sa.String(), unique=True, nullable=False),
        sa.Column("graph_database_name", sa.String(), unique=True, nullable=False),
        sa.Column("vector_database_provider", sa.String(), unique=False, nullable=False),
        sa.Column("graph_database_provider", sa.String(), unique=False, nullable=False),
        sa.Column("vector_database_url", sa.String(), unique=False, nullable=True),
        sa.Column("graph_database_url", sa.String(), unique=False, nullable=True),
        sa.Column("vector_database_key", sa.String(), unique=False, nullable=True),
        sa.Column("graph_database_key", sa.String(), unique=False, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )

    vector_database_provider_column = _get_column(
        insp, "dataset_database", "vector_database_provider"
    )
    if not vector_database_provider_column:
        op.add_column(
            "dataset_database",
            sa.Column("vector_database_provider", sa.String(), unique=False, nullable=False),
        )
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table("dataset_database") as batch_op:
                batch_op.execute(
                    data.update().values(
                        vector_database_provider="lancedb",
                    )
                )
        else:
            conn = op.get_bind()
            conn.execute(data.update().values(vector_database_provider="lancedb"))

    graph_database_provider_column = _get_column(
        insp, "dataset_database", "graph_database_provider"
    )
    if not graph_database_provider_column:
        op.add_column(
            "dataset_database",
            sa.Column("graph_database_provider", sa.String(), unique=False, nullable=False),
        )
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table("dataset_database") as batch_op:
                batch_op.execute(
                    data.update().values(
                        graph_database_provider="kuzu",
                    )
                )
        else:
            conn = op.get_bind()
            conn.execute(data.update().values(graph_database_provider="kuzu"))

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
