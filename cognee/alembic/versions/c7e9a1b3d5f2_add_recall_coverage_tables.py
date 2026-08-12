"""add_recall_coverage_tables

Revision ID: c7e9a1b3d5f2
Revises: e5a7b9c1d3f4
Create Date: 2026-08-12 00:00:00.000000

Creates the five recall-coverage tables: runs, judged question rows, the
owner-scoped topic taxonomy, its pending suggestions, and human-curated
questions.

Statuses and scopes are plain strings, not native enums, so adding a value later
is an application change rather than a raw-DDL migration (compare
``1d0bb7fede17_add_pipeline_run_status.py``). Owner/user/dataset ids are bare
indexed UUIDs with no foreign keys, matching ``queries``.

Every step is guarded by an inspector check: ``Base.metadata.create_all()`` may
already have created these tables on a fresh database before migrations run.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7e9a1b3d5f2"
down_revision: Union[str, None] = "a7c3e9f1b5d8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES = (
    "recall_coverage_runs",
    "recall_coverage_questions",
    "recall_coverage_topics",
    "recall_coverage_topic_suggestions",
    "recall_coverage_curated_questions",
)


def _create_runs() -> None:
    op.create_table(
        "recall_coverage_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("agent_label", sa.String(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recall_row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("distinct_ask_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("collapsed_retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("question_row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("curated_question_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("topic_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("dataset_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("user_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("taxonomy_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recall_coverage_runs_agent_label", "recall_coverage_runs", ["agent_label"])
    op.create_index("ix_recall_coverage_runs_owner_id", "recall_coverage_runs", ["owner_id"])
    op.create_index("ix_recall_coverage_runs_status", "recall_coverage_runs", ["status"])
    op.create_index("ix_recall_coverage_runs_created_at", "recall_coverage_runs", ["created_at"])


def _create_questions() -> None:
    op.create_table(
        "recall_coverage_questions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.UUID(), nullable=False),
        sa.Column("question_group_id", sa.UUID(), nullable=True),
        sa.Column("user_id", sa.UUID(), nullable=False),
        sa.Column("dataset_id", sa.UUID(), nullable=True),
        sa.Column("dataset_name", sa.String(), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("source", sa.String(), nullable=False, server_default="observed"),
        sa.Column("was_asked", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("curated_question_id", sa.UUID(), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("judge_score", sa.Integer(), nullable=True),
        sa.Column("judge_answered", sa.Boolean(), nullable=True),
        sa.Column("retrieval_context", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("topic_id", sa.UUID(), nullable=True),
        sa.Column("first_asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_asked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("occurrence_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("impact", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recall_coverage_questions_run_id", "recall_coverage_questions", ["run_id"])
    op.create_index(
        "ix_recall_coverage_questions_question_group_id",
        "recall_coverage_questions",
        ["question_group_id"],
    )
    op.create_index(
        "ix_recall_coverage_questions_user_id", "recall_coverage_questions", ["user_id"]
    )
    op.create_index(
        "ix_recall_coverage_questions_dataset_id", "recall_coverage_questions", ["dataset_id"]
    )
    op.create_index(
        "ix_recall_coverage_questions_topic_id", "recall_coverage_questions", ["topic_id"]
    )


def _create_topics() -> None:
    op.create_table(
        "recall_coverage_topics",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("centroid", sa.JSON(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("seed_question_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("taxonomy_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_recall_coverage_topics_owner_id", "recall_coverage_topics", ["owner_id"])
    op.create_index(
        "ix_recall_coverage_topics_owner_deleted",
        "recall_coverage_topics",
        ["owner_id", "deleted_at"],
    )


def _create_topic_suggestions() -> None:
    op.create_table(
        "recall_coverage_topic_suggestions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("agent_label", sa.String(), nullable=True),
        sa.Column("run_id", sa.UUID(), nullable=True),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("centroid", sa.JSON(), nullable=False),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column("question_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cohesion", sa.Float(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("accepted_topic_id", sa.UUID(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recall_coverage_topic_suggestions_owner_id",
        "recall_coverage_topic_suggestions",
        ["owner_id"],
    )
    op.create_index(
        "ix_recall_coverage_topic_suggestions_run_id",
        "recall_coverage_topic_suggestions",
        ["run_id"],
    )
    op.create_index(
        "ix_recall_coverage_topic_suggestions_status",
        "recall_coverage_topic_suggestions",
        ["status"],
    )


def _create_curated_questions() -> None:
    op.create_table(
        "recall_coverage_curated_questions",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("owner_id", sa.UUID(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False, server_default="agent"),
        sa.Column("agent_label", sa.String(), nullable=True),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_recall_coverage_curated_questions_owner_id",
        "recall_coverage_curated_questions",
        ["owner_id"],
    )
    op.create_index(
        "ix_recall_coverage_curated_questions_owner_scope",
        "recall_coverage_curated_questions",
        ["owner_id", "scope"],
    )


_CREATORS = {
    "recall_coverage_runs": _create_runs,
    "recall_coverage_questions": _create_questions,
    "recall_coverage_topics": _create_topics,
    "recall_coverage_topic_suggestions": _create_topic_suggestions,
    "recall_coverage_curated_questions": _create_curated_questions,
}


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    existing_tables = set(insp.get_table_names())

    for table in _TABLES:
        if table in existing_tables:
            continue
        _CREATORS[table]()


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    existing_tables = set(insp.get_table_names())

    for table in reversed(_TABLES):
        if table in existing_tables:
            op.drop_table(table)
