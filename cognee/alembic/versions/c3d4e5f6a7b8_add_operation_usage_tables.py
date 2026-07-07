"""Add operation usage tables.

Revision ID: c3d4e5f6a7b8
Revises: aa753a730673
Create Date: 2026-06-24 10:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "aa753a730673"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OPERATION_USAGE_RECORDS_TABLE = "operation_usage_records"
OPERATION_MODEL_USAGE_TABLE = "operation_model_usage"

OPERATION_USAGE_RECORDS_INDEXES = (
    ("ix_operation_usage_records_user_id", ["user_id"]),
    ("ix_operation_usage_records_operation_type", ["operation_type"]),
    ("ix_operation_usage_records_dataset_id", ["dataset_id"]),
    ("ix_operation_usage_records_status", ["status"]),
    ("ix_operation_usage_records_last_activity_at", ["last_activity_at"]),
)
OPERATION_MODEL_USAGE_INDEXES = (("ix_operation_model_usage_user_id", ["user_id"]),)


def _get_table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    table_names = _get_table_names()

    if OPERATION_USAGE_RECORDS_TABLE not in table_names:
        op.create_table(
            OPERATION_USAGE_RECORDS_TABLE,
            sa.Column("operation_id", sa.String(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("operation_type", sa.String(), nullable=False),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="running"),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
            sa.Column("last_model", sa.Text(), nullable=True),
        )
        for index_name, columns in OPERATION_USAGE_RECORDS_INDEXES:
            op.create_index(index_name, OPERATION_USAGE_RECORDS_TABLE, columns, unique=False)
    else:
        print(f"{OPERATION_USAGE_RECORDS_TABLE} table already exists, skipping creation")

    if OPERATION_MODEL_USAGE_TABLE not in table_names:
        op.create_table(
            OPERATION_MODEL_USAGE_TABLE,
            sa.Column("operation_id", sa.String(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("model", sa.Text(), nullable=False, primary_key=True),
            sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        for index_name, columns in OPERATION_MODEL_USAGE_INDEXES:
            op.create_index(index_name, OPERATION_MODEL_USAGE_TABLE, columns, unique=False)
    else:
        print(f"{OPERATION_MODEL_USAGE_TABLE} table already exists, skipping creation")


def downgrade() -> None:
    table_names = _get_table_names()

    if OPERATION_MODEL_USAGE_TABLE in table_names:
        for index_name, _ in OPERATION_MODEL_USAGE_INDEXES:
            op.drop_index(index_name, table_name=OPERATION_MODEL_USAGE_TABLE)
        op.drop_table(OPERATION_MODEL_USAGE_TABLE)
    else:
        print(f"{OPERATION_MODEL_USAGE_TABLE} table doesn't exist, skipping downgrade")

    if OPERATION_USAGE_RECORDS_TABLE in table_names:
        for index_name, _ in OPERATION_USAGE_RECORDS_INDEXES:
            op.drop_index(index_name, table_name=OPERATION_USAGE_RECORDS_TABLE)
        op.drop_table(OPERATION_USAGE_RECORDS_TABLE)
    else:
        print(f"{OPERATION_USAGE_RECORDS_TABLE} table doesn't exist, skipping downgrade")
