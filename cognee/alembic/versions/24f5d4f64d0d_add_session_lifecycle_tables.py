"""Add session lifecycle tables.

Revision ID: 24f5d4f64d0d
Revises: 7c5d4e2f8a91
Create Date: 2026-04-24 16:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "24f5d4f64d0d"
down_revision: Union[str, None] = "7c5d4e2f8a91"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SESSION_RECORDS_TABLE = "session_records"
SESSION_MODEL_USAGE_TABLE = "session_model_usage"

SESSION_RECORDS_INDEXES = (
    ("ix_session_records_user_id", ["user_id"]),
    ("ix_session_records_dataset_id", ["dataset_id"]),
    ("ix_session_records_status", ["status"]),
    ("ix_session_records_last_activity_at", ["last_activity_at"]),
)
SESSION_MODEL_USAGE_INDEXES = (("ix_session_model_usage_user_id", ["user_id"]),)


def _get_table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    table_names = _get_table_names()

    if SESSION_RECORDS_TABLE not in table_names:
        op.create_table(
            SESSION_RECORDS_TABLE,
            sa.Column("session_id", sa.String(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("status", sa.String(), nullable=False, server_default="running"),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_activity_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
            sa.Column("error_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_model", sa.Text(), nullable=True),
        )
        for index_name, columns in SESSION_RECORDS_INDEXES:
            op.create_index(index_name, SESSION_RECORDS_TABLE, columns, unique=False)
    else:
        print(f"{SESSION_RECORDS_TABLE} table already exists, skipping creation")

    if SESSION_MODEL_USAGE_TABLE not in table_names:
        op.create_table(
            SESSION_MODEL_USAGE_TABLE,
            sa.Column("session_id", sa.String(), nullable=False, primary_key=True),
            sa.Column("user_id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("model", sa.Text(), nullable=False, primary_key=True),
            sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        for index_name, columns in SESSION_MODEL_USAGE_INDEXES:
            op.create_index(index_name, SESSION_MODEL_USAGE_TABLE, columns, unique=False)
    else:
        print(f"{SESSION_MODEL_USAGE_TABLE} table already exists, skipping creation")


def downgrade() -> None:
    table_names = _get_table_names()

    if SESSION_MODEL_USAGE_TABLE in table_names:
        for index_name, _ in SESSION_MODEL_USAGE_INDEXES:
            op.drop_index(index_name, table_name=SESSION_MODEL_USAGE_TABLE)
        op.drop_table(SESSION_MODEL_USAGE_TABLE)
    else:
        print(f"{SESSION_MODEL_USAGE_TABLE} table doesn't exist, skipping downgrade")

    if SESSION_RECORDS_TABLE in table_names:
        for index_name, _ in SESSION_RECORDS_INDEXES:
            op.drop_index(index_name, table_name=SESSION_RECORDS_TABLE)
        op.drop_table(SESSION_RECORDS_TABLE)
    else:
        print(f"{SESSION_RECORDS_TABLE} table doesn't exist, skipping downgrade")
