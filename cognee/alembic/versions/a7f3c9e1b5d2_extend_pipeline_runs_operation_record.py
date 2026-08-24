"""Extend pipeline_runs into the operation record (SDK-399)

Revision ID: a7f3c9e1b5d2
Revises: b8c1d3e5f7a9
Create Date: 2026-08-17 00:00:00.000000

Adds nullable operation-record columns to pipeline_runs so it can serve as
the durable record of ALL cognee operations (pipeline-backed and
non-pipeline alike): triggering user/tenant, operation name, start/end
timestamps, succeeded/failed outcome with error class and a PII-scrubbed
error message, token spend, originating surface, session linkage, parent
operation linkage, and a background-launch flag.
Old rows stay NULL — no backfill. No enum changes (PipelineRunStatus is
frozen; outcome is a plain String).

Idempotent: columns may already exist on fresh databases where
Base.metadata.create_all() ran before alembic stamped head, so every
add/drop is inspector-guarded.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7f3c9e1b5d2"
down_revision: Union[str, None] = "b8c1d3e5f7a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "pipeline_runs"

# (column name, SQLAlchemy type factory)
_NEW_COLUMNS = [
    ("user_id", lambda: sa.UUID()),
    ("tenant_id", lambda: sa.UUID()),
    ("operation_name", lambda: sa.String()),
    ("started_at", lambda: sa.DateTime(timezone=True)),
    ("ended_at", lambda: sa.DateTime(timezone=True)),
    ("outcome", lambda: sa.String()),
    ("error_class", lambda: sa.String()),
    ("tokens_in", lambda: sa.Integer()),
    ("tokens_out", lambda: sa.Integer()),
    ("origin", lambda: sa.String()),
    ("session_id", lambda: sa.String()),
    ("parent_operation_id", lambda: sa.UUID()),
    ("background", lambda: sa.Boolean()),
    ("error_message", lambda: sa.String()),
]

# (index name, column name)
_NEW_INDEXES = [
    ("ix_pipeline_runs_user_id", "user_id"),
    ("ix_pipeline_runs_operation_name", "operation_name"),
    ("ix_pipeline_runs_outcome", "outcome"),
    ("ix_pipeline_runs_session_id", "session_id"),
    ("ix_pipeline_runs_parent_operation_id", "parent_operation_id"),
]


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def _index_names(inspector, table):
    return {index["name"] for index in inspector.get_indexes(table)}


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    missing = [
        (name, make_type)
        for name, make_type in _NEW_COLUMNS
        if not _get_column(insp, TABLE_NAME, name)
    ]

    if missing:
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table(TABLE_NAME) as batch_op:
                for name, make_type in missing:
                    batch_op.add_column(sa.Column(name, make_type(), nullable=True))
        else:
            for name, make_type in missing:
                op.add_column(TABLE_NAME, sa.Column(name, make_type(), nullable=True))

    insp = sa.inspect(conn)
    existing_indexes = _index_names(insp, TABLE_NAME)
    for index_name, column_name in _NEW_INDEXES:
        if index_name not in existing_indexes:
            op.create_index(index_name, TABLE_NAME, [column_name])


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    existing_indexes = _index_names(insp, TABLE_NAME)
    for index_name, _column_name in _NEW_INDEXES:
        if index_name in existing_indexes:
            op.drop_index(index_name, table_name=TABLE_NAME)

    insp = sa.inspect(conn)
    present = [name for name, _make_type in _NEW_COLUMNS if _get_column(insp, TABLE_NAME, name)]

    if present:
        if op.get_context().dialect.name == "sqlite":
            with op.batch_alter_table(TABLE_NAME) as batch_op:
                for name in present:
                    batch_op.drop_column(name)
        else:
            for name in present:
                op.drop_column(TABLE_NAME, name)
