"""Add memory score tables.

Revision ID: d4f6a8b0c2e3
Revises: c3d5e7f9a1b2
Create Date: 2026-08-04 10:00:00.000000

Creates the two tables backing the tenant memory-accuracy score:
``memory_score_runs`` (one row per evaluation run) and
``memory_score_questions`` (one row per answered + judged question).

Both are idempotent-guarded so a database that already has the tables
(created by ``Base.metadata.create_all`` on the fresh-database path)
is left untouched.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d4f6a8b0c2e3"
down_revision: Union[str, None] = "c3d5e7f9a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RUNS_TABLE = "memory_score_runs"
QUESTIONS_TABLE = "memory_score_questions"

# Name must match what SQLAlchemy derives from the MemoryScoreRunStatus
# class name, otherwise autogenerate reports a permanent diff.
STATUS_ENUM_NAME = "memoryscorerunstatus"
STATUS_VALUES = (
    "INITIATED",
    "RUNNING",
    "COMPLETED",
    "ERRORED",
    "SKIPPED_INSUFFICIENT_DATA",
)

RUNS_INDEXES = (
    ("ix_memory_score_runs_tenant_id", ["tenant_id"]),
    ("ix_memory_score_runs_dataset_id", ["dataset_id"]),
    ("ix_memory_score_runs_created_at", ["created_at"]),
    ("ix_memory_score_runs_tenant_id_created_at", ["tenant_id", "created_at"]),
)
QUESTIONS_INDEXES = (("ix_memory_score_questions_run_id", ["run_id"]),)


def _get_table_names() -> set[str]:
    return set(sa.inspect(op.get_bind()).get_table_names())


def upgrade() -> None:
    table_names = _get_table_names()

    if RUNS_TABLE not in table_names:
        op.create_table(
            RUNS_TABLE,
            sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("tenant_id", sa.UUID(), nullable=True),
            sa.Column("dataset_id", sa.UUID(), nullable=True),
            sa.Column("triggered_by_user_id", sa.UUID(), nullable=True),
            sa.Column(
                "status",
                sa.Enum(*STATUS_VALUES, name=STATUS_ENUM_NAME),
                nullable=True,
            ),
            sa.Column("below_data_floor", sa.Boolean(), nullable=True, server_default=sa.false()),
            sa.Column("floor_reason", sa.String(), nullable=True),
            sa.Column("schema_defined", sa.Boolean(), nullable=True, server_default=sa.false()),
            sa.Column("overall_accuracy", sa.Float(), nullable=True),
            sa.Column("synthetic_question_count", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("real_question_count", sa.Integer(), nullable=True, server_default="0"),
            sa.Column("topics", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )
        for index_name, columns in RUNS_INDEXES:
            op.create_index(index_name, RUNS_TABLE, columns, unique=False)
    else:
        print(f"{RUNS_TABLE} table already exists, skipping creation")

    if QUESTIONS_TABLE not in table_names:
        op.create_table(
            QUESTIONS_TABLE,
            sa.Column("id", sa.UUID(), nullable=False, primary_key=True),
            sa.Column("run_id", sa.UUID(), nullable=True),
            sa.Column("topic", sa.String(), nullable=True),
            sa.Column("source", sa.String(), nullable=True),
            sa.Column("text", sa.Text(), nullable=True),
            sa.Column("expected_answer", sa.Text(), nullable=True),
            sa.Column("answer", sa.Text(), nullable=True),
            sa.Column("score", sa.Float(), nullable=True),
            sa.Column("grounded", sa.Boolean(), nullable=True),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("source_query_id", sa.UUID(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        for index_name, columns in QUESTIONS_INDEXES:
            op.create_index(index_name, QUESTIONS_TABLE, columns, unique=False)
    else:
        print(f"{QUESTIONS_TABLE} table already exists, skipping creation")


def downgrade() -> None:
    table_names = _get_table_names()

    if QUESTIONS_TABLE in table_names:
        for index_name, _ in QUESTIONS_INDEXES:
            op.drop_index(index_name, table_name=QUESTIONS_TABLE)
        op.drop_table(QUESTIONS_TABLE)
    else:
        print(f"{QUESTIONS_TABLE} table doesn't exist, skipping downgrade")

    if RUNS_TABLE in table_names:
        for index_name, _ in RUNS_INDEXES:
            op.drop_index(index_name, table_name=RUNS_TABLE)
        op.drop_table(RUNS_TABLE)
        # Postgres keeps the enum type around after the table is gone.
        bind = op.get_bind()
        if bind.dialect.name == "postgresql":
            sa.Enum(*STATUS_VALUES, name=STATUS_ENUM_NAME).drop(bind, checkfirst=True)
    else:
        print(f"{RUNS_TABLE} table doesn't exist, skipping downgrade")
