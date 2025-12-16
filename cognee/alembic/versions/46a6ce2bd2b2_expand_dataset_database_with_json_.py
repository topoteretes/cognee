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

graph_constraint_name = "dataset_database_graph_database_name_key"
vector_constraint_name = "dataset_database_vector_database_name_key"
TABLE_NAME = "dataset_database"


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def _recreate_table_without_unique_constraint_sqlite(op, insp):
    """
    SQLite cannot drop unique constraints on individual columns. We must:
    1. Create a new table without the unique constraints.
    2. Copy data from the old table.
    3. Drop the old table.
    4. Rename the new table.
    """
    conn = op.get_bind()

    # Create new table definition (without unique constraints)
    op.create_table(
        f"{TABLE_NAME}_new",
        sa.Column("owner_id", sa.UUID()),
        sa.Column("dataset_id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("vector_database_name", sa.String(), nullable=False),
        sa.Column("graph_database_name", sa.String(), nullable=False),
        sa.Column("vector_database_provider", sa.String(), nullable=False),
        sa.Column("graph_database_provider", sa.String(), nullable=False),
        sa.Column(
            "vector_dataset_database_handler",
            sa.String(),
            unique=False,
            nullable=False,
            server_default="lancedb",
        ),
        sa.Column(
            "graph_dataset_database_handler",
            sa.String(),
            unique=False,
            nullable=False,
            server_default="kuzu",
        ),
        sa.Column("vector_database_url", sa.String()),
        sa.Column("graph_database_url", sa.String()),
        sa.Column("vector_database_key", sa.String()),
        sa.Column("graph_database_key", sa.String()),
        sa.Column(
            "graph_database_connection_info",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "vector_database_connection_info",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["principals.id"], ondelete="CASCADE"),
    )

    # Copy data into new table
    conn.execute(
        sa.text(f"""
        INSERT INTO {TABLE_NAME}_new
        SELECT
            owner_id,
            dataset_id,
            vector_database_name,
            graph_database_name,
            vector_database_provider,
            graph_database_provider,
            vector_dataset_database_handler,
            graph_dataset_database_handler,
            vector_database_url,
            graph_database_url,
            vector_database_key,
            graph_database_key,
            COALESCE(graph_database_connection_info, '{{}}'),
            COALESCE(vector_database_connection_info, '{{}}'),
            created_at,
            updated_at
        FROM {TABLE_NAME}
    """)
    )

    # Drop old table
    op.drop_table(TABLE_NAME)

    # Rename new table
    op.rename_table(f"{TABLE_NAME}_new", TABLE_NAME)


def _recreate_table_with_unique_constraint_sqlite(op, insp):
    """
    SQLite cannot drop unique constraints on individual columns. We must:
    1. Create a new table without the unique constraints.
    2. Copy data from the old table.
    3. Drop the old table.
    4. Rename the new table.
    """
    conn = op.get_bind()

    # Create new table definition (without unique constraints)
    op.create_table(
        f"{TABLE_NAME}_new",
        sa.Column("owner_id", sa.UUID()),
        sa.Column("dataset_id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("vector_database_name", sa.String(), nullable=False, unique=True),
        sa.Column("graph_database_name", sa.String(), nullable=False, unique=True),
        sa.Column("vector_database_provider", sa.String(), nullable=False),
        sa.Column("graph_database_provider", sa.String(), nullable=False),
        sa.Column(
            "vector_dataset_database_handler",
            sa.String(),
            unique=False,
            nullable=False,
            server_default="lancedb",
        ),
        sa.Column(
            "graph_dataset_database_handler",
            sa.String(),
            unique=False,
            nullable=False,
            server_default="kuzu",
        ),
        sa.Column("vector_database_url", sa.String()),
        sa.Column("graph_database_url", sa.String()),
        sa.Column("vector_database_key", sa.String()),
        sa.Column("graph_database_key", sa.String()),
        sa.Column(
            "graph_database_connection_info",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "vector_database_connection_info",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["owner_id"], ["principals.id"], ondelete="CASCADE"),
    )

    # Copy data into new table
    conn.execute(
        sa.text(f"""
        INSERT INTO {TABLE_NAME}_new
        SELECT
            owner_id,
            dataset_id,
            vector_database_name,
            graph_database_name,
            vector_database_provider,
            graph_database_provider,
            vector_dataset_database_handler,
            graph_dataset_database_handler,
            vector_database_url,
            graph_database_url,
            vector_database_key,
            graph_database_key,
            COALESCE(graph_database_connection_info, '{{}}'),
            COALESCE(vector_database_connection_info, '{{}}'),
            created_at,
            updated_at
        FROM {TABLE_NAME}
    """)
    )

    # Drop old table
    op.drop_table(TABLE_NAME)

    # Rename new table
    op.rename_table(f"{TABLE_NAME}_new", TABLE_NAME)


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    unique_constraints = insp.get_unique_constraints(TABLE_NAME)

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
                server_default=sa.text("'{}'"),
            ),
        )

    vector_dataset_database_handler = _get_column(
        insp, "dataset_database", "vector_dataset_database_handler"
    )
    if not vector_dataset_database_handler:
        # Add LanceDB as the default graph dataset database handler
        op.add_column(
            "dataset_database",
            sa.Column(
                "vector_dataset_database_handler",
                sa.String(),
                unique=False,
                nullable=False,
                server_default="lancedb",
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
                server_default=sa.text("'{}'"),
            ),
        )

    graph_dataset_database_handler = _get_column(
        insp, "dataset_database", "graph_dataset_database_handler"
    )
    if not graph_dataset_database_handler:
        # Add Kuzu as the default graph dataset database handler
        op.add_column(
            "dataset_database",
            sa.Column(
                "graph_dataset_database_handler",
                sa.String(),
                unique=False,
                nullable=False,
                server_default="kuzu",
            ),
        )

    with op.batch_alter_table("dataset_database", schema=None) as batch_op:
        # Drop the unique constraint to make unique=False
        graph_constraint_to_drop = None
        for uc in unique_constraints:
            # Check if the constraint covers ONLY the target column
            if uc["name"] == graph_constraint_name:
                graph_constraint_to_drop = uc["name"]
                break

        vector_constraint_to_drop = None
        for uc in unique_constraints:
            # Check if the constraint covers ONLY the target column
            if uc["name"] == vector_constraint_name:
                vector_constraint_to_drop = uc["name"]
                break

        if (
            vector_constraint_to_drop
            and graph_constraint_to_drop
            and op.get_context().dialect.name == "postgresql"
        ):
            # PostgreSQL
            batch_op.drop_constraint(graph_constraint_name, type_="unique")
            batch_op.drop_constraint(vector_constraint_name, type_="unique")

        if op.get_context().dialect.name == "sqlite":
            conn = op.get_bind()
            # Fun fact: SQLite has hidden auto indexes for unique constraints that can't be dropped or accessed directly
            #           So we need to check for them and drop them by recreating the table (altering column also won't work)
            result = conn.execute(sa.text("PRAGMA index_list('dataset_database')"))
            rows = result.fetchall()
            unique_auto_indexes = [row for row in rows if row[3] == "u"]
            for row in unique_auto_indexes:
                result = conn.execute(sa.text(f"PRAGMA index_info('{row[1]}')"))
                index_info = result.fetchall()
                if index_info[0][2] == "vector_database_name":
                    # In case a unique index exists on vector_database_name, drop it and the graph_database_name one
                    _recreate_table_without_unique_constraint_sqlite(op, insp)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if op.get_context().dialect.name == "sqlite":
        _recreate_table_with_unique_constraint_sqlite(op, insp)
    elif op.get_context().dialect.name == "postgresql":
        with op.batch_alter_table("dataset_database", schema=None) as batch_op:
            # Re-add the unique constraint to return to unique=True
            batch_op.create_unique_constraint(graph_constraint_name, ["graph_database_name"])

        with op.batch_alter_table("dataset_database", schema=None) as batch_op:
            # Re-add the unique constraint to return to unique=True
            batch_op.create_unique_constraint(vector_constraint_name, ["vector_database_name"])

    op.drop_column("dataset_database", "vector_database_connection_info")
    op.drop_column("dataset_database", "graph_database_connection_info")
    op.drop_column("dataset_database", "vector_dataset_database_handler")
    op.drop_column("dataset_database", "graph_dataset_database_handler")
