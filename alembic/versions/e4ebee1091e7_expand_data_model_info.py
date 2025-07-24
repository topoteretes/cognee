"""Expand data model info

Revision ID: e4ebee1091e7
Revises: ab7e313804ae
Create Date: 2025-07-24 13:21:30.738486

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e4ebee1091e7"
down_revision: Union[str, None] = "ab7e313804ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def _index_exists(inspector, table, name, schema=None):
    return any(ix["name"] == name for ix in inspector.get_indexes(table, schema=schema))


def upgrade() -> None:
    TABLES_TO_DROP = [
        "file_metadata",
        "_dlt_loads",
        "_dlt_version",
        "_dlt_pipeline_state",
    ]

    conn = op.get_bind()
    insp = sa.inspect(conn)
    existing = set(insp.get_table_names())

    for tbl in TABLES_TO_DROP:
        if tbl in existing:
            op.drop_table(tbl)

    DATA_TABLE = "data"
    DATA_TENANT_COL = "tenant_id"
    DATA_SIZE_COL = "data_size"
    DATA_TENANT_IDX = "ix_data_tenant_id"

    # --- tenant_id ---
    col = _get_column(insp, DATA_TABLE, DATA_TENANT_COL)
    if col is None:
        op.add_column(
            DATA_TABLE,
            sa.Column(DATA_TENANT_COL, postgresql.UUID(as_uuid=True), nullable=True),
        )
    else:
        # Column exists – fix nullability if needed
        if col.get("nullable", True) is False:
            op.alter_column(
                DATA_TABLE,
                DATA_TENANT_COL,
                existing_type=postgresql.UUID(as_uuid=True),
                nullable=True,
            )

    # --- data_size ---
    col = _get_column(insp, DATA_TABLE, DATA_SIZE_COL)
    if col is None:
        op.add_column(DATA_TABLE, sa.Column(DATA_SIZE_COL, sa.Integer(), nullable=True))
    else:
        # If you also need to change nullability for data_size, do it here
        if col.get("nullable", True) is False:
            op.alter_column(
                DATA_TABLE,
                DATA_SIZE_COL,
                existing_type=sa.Integer(),
                nullable=True,
            )

    # --- index on tenant_id ---
    if not _index_exists(insp, DATA_TABLE, DATA_TENANT_IDX):
        op.create_index(DATA_TENANT_IDX, DATA_TABLE, [DATA_TENANT_COL], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_data_tenant_id"), table_name="data")
    op.drop_column("data", "data_size")
    op.drop_column("data", "tenant_id")
    op.create_table(
        "_dlt_pipeline_state",
        sa.Column("version", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column("engine_version", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column("pipeline_name", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("state", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column(
            "created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("version_hash", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("_dlt_load_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("_dlt_id", sa.VARCHAR(length=128), autoincrement=False, nullable=False),
    )
    op.create_table(
        "_dlt_version",
        sa.Column("version", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column("engine_version", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column(
            "inserted_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("schema_name", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("version_hash", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("schema", sa.TEXT(), autoincrement=False, nullable=False),
    )
    op.create_table(
        "_dlt_loads",
        sa.Column("load_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("schema_name", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("status", sa.BIGINT(), autoincrement=False, nullable=False),
        sa.Column(
            "inserted_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=False
        ),
        sa.Column("schema_version_hash", sa.TEXT(), autoincrement=False, nullable=True),
    )
    op.create_table(
        "file_metadata",
        sa.Column("id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("name", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("file_path", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("extension", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("mime_type", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("content_hash", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("owner_id", sa.TEXT(), autoincrement=False, nullable=True),
        sa.Column("_dlt_load_id", sa.TEXT(), autoincrement=False, nullable=False),
        sa.Column("_dlt_id", sa.VARCHAR(length=128), autoincrement=False, nullable=False),
        sa.Column("node_set", sa.TEXT(), autoincrement=False, nullable=True),
    )
