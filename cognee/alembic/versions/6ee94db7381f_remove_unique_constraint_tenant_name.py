"""remove unique constraint tenant name

Revision ID: 6ee94db7381f
Revises: d4e5f6a7b8c9
Create Date: 2026-04-13 18:30:04.942834

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6ee94db7381f"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _unique_constraint_name(inspector: sa.Inspector, table: str, column: str) -> str | None:
    for constraint in inspector.get_unique_constraints(table):
        column_names = constraint.get("column_names") or []
        if len(column_names) == 1 and column_names[0] == column:
            return constraint.get("name")
    return None


def _index_name_for_column(
    inspector: sa.Inspector, table: str, column: str, *, unique: bool | None = None
) -> str | None:
    for index in inspector.get_indexes(table):
        column_names = index.get("column_names") or []
        if len(column_names) == 1 and column_names[0] == column:
            if unique is None or bool(index.get("unique")) is unique:
                return index.get("name")
    return None


def _has_unique_on_column(inspector: sa.Inspector, table: str, column: str) -> bool:
    return (
        _unique_constraint_name(inspector, table, column) is not None
        or _index_name_for_column(inspector, table, column, unique=True) is not None
    )


def _recreate_tenants_table_sqlite(unique_name: bool) -> None:
    conn = op.get_bind()

    op.create_table(
        "tenants_new",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=True),
        sa.ForeignKeyConstraint(["id"], ["principals.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    conn.execute(
        sa.text(
            "INSERT INTO tenants_new (id, name, owner_id) SELECT id, name, owner_id FROM tenants"
        )
    )

    op.drop_table("tenants")
    op.rename_table("tenants_new", "tenants")

    op.create_index("ix_tenants_owner_id", "tenants", ["owner_id"], unique=False)
    op.create_index("ix_tenants_name", "tenants", ["name"], unique=unique_name)


def upgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        conn = op.get_bind()

        result = conn.execute(sa.text("PRAGMA index_list('tenants')"))
        rows = result.fetchall()
        # Check if there's a unique index on the name column. If third element in row is 1, it's a unique index
        unique_auto_indexes = [row for row in rows if row[2] == 1]
        for row in unique_auto_indexes:
            result = conn.execute(sa.text(f"PRAGMA index_info('{row[1]}')"))
            index_info = result.fetchall()
            if index_info[0][2] == "name":
                # In case a unique index exists on tenant name remove uniqueness
                _recreate_tenants_table_sqlite(unique_name=False)
        return

    conn = op.get_bind()
    inspector = sa.inspect(conn)

    unique_constraint = _unique_constraint_name(inspector, "tenants", "name")

    if unique_constraint:
        with op.batch_alter_table("tenants", schema=None) as batch_op:
            batch_op.drop_constraint(unique_constraint, type_="unique")

    # Re-inspect after potential constraint drop to avoid stale metadata.
    inspector = sa.inspect(conn)
    unique_index = _index_name_for_column(inspector, "tenants", "name", unique=True)
    if unique_index:
        op.drop_index(unique_index, table_name="tenants")
        op.create_index(unique_index, "tenants", ["name"], unique=False)

    # Keep name indexed after removing uniqueness.
    inspector = sa.inspect(conn)
    if _index_name_for_column(inspector, "tenants", "name", unique=None) is None:
        op.create_index("ix_tenants_name", "tenants", ["name"], unique=False)


def downgrade() -> None:
    if op.get_context().dialect.name == "sqlite":
        _recreate_tenants_table_sqlite(unique_name=True)
        return

    conn = op.get_bind()
    inspector = sa.inspect(conn)

    if _has_unique_on_column(inspector, "tenants", "name"):
        return

    existing_name_index = _index_name_for_column(inspector, "tenants", "name", unique=False)
    if existing_name_index:
        op.drop_index(existing_name_index, table_name="tenants")
        op.create_index(existing_name_index, "tenants", ["name"], unique=True)
    else:
        with op.batch_alter_table("tenants", schema=None) as batch_op:
            batch_op.create_unique_constraint("tenants_name_key", ["name"])
