"""Index pipeline_runs on (created_at, id)

Revision ID: c4e8a1f6b3d7
Revises: c7e2a9b4d1f3
Create Date: 2026-08-21 00:00:00.000000

Since a7f3c9e1b5d2 (SDK-399) pipeline_runs holds a row per *operation*, not
just per pipeline run, so it grows far faster than before. Every other column
the activity feed and the time-bucketed usage queries touch was already
indexed (user_id, dataset_id, operation_name, outcome, session_id,
parent_operation_id, pipeline_run_id, pipeline_id) — created_at was not, even
though those reads all order or range-scan by it.

Composite rather than created_at alone: it matches the feed's
``ORDER BY created_at DESC, id DESC`` exactly (id is the pagination
tiebreaker, without which OFFSET paging re-serves rows sharing a timestamp)
and serves a created_at range scan just as well.

Plain CREATE INDEX, not CONCURRENTLY: CONCURRENTLY cannot run inside
Alembic's transaction, and migrations run before the app serves traffic, so
the brief write lock is uncontended.

Idempotent and inspector-guarded, matching a7f3c9e1b5d2: the index already
exists on fresh databases, where Base.metadata.create_all() builds the schema
at head (PipelineRun declares this index in __table_args__) before alembic
stamps head.

A missing pipeline_runs table is skipped quietly (create_all simply has not
run yet — same as a7f3c9e1b5d2 and e5a7b9c1d3f4), but a pipeline_runs table
missing created_at or id raises: see _require_index_columns.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c4e8a1f6b3d7"
down_revision: Union[str, None] = "c7e2a9b4d1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "pipeline_runs"
INDEX_NAME = "ix_pipeline_runs_created_at_id"
INDEX_COLUMNS = ["created_at", "id"]


def _index_names(inspector, table):
    return {index["name"] for index in inspector.get_indexes(table)}


def _require_index_columns(inspector) -> None:
    """Fail loudly when pipeline_runs lacks the columns being indexed.

    Raises rather than skipping, deliberately. No migration has ever created
    pipeline_runs — Base.metadata.create_all() is its only creator, and the
    initial revision is a no-op — so created_at and id come into existence
    with the table itself, and a7f3c9e1b5d2 only ADDS other columns. Alembic
    also runs revisions strictly in down_revision order, so a table without
    these columns is not a state this chain can reach, not even half-applied:
    it is a foreign or damaged pipeline_runs.

    Skipping would be worse than failing. Alembic stamps c4e8a1f6b3d7 on the
    way out either way, permanently recording the index as present; nothing
    re-examines it later, so the slow scans this revision exists to remove
    would stay, invisibly. Failing here keeps the stamp at a7f3c9e1b5d2 and
    names the problem while it is still fixable.
    """
    columns = {column["name"] for column in inspector.get_columns(TABLE_NAME)}
    missing = [name for name in INDEX_COLUMNS if name not in columns]

    if missing:
        raise RuntimeError(
            f"Cannot create {INDEX_NAME}: table {TABLE_NAME} is missing "
            f"{', '.join(missing)}. Alembic never creates or drops these columns, "
            f"so this database's {TABLE_NAME} is not the one cognee defines. "
            f"Reconcile the table with cognee.modules.pipelines.models.PipelineRun, "
            f"then re-run the migration."
        )


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if INDEX_NAME not in _index_names(insp, TABLE_NAME):
        _require_index_columns(insp)
        op.create_index(INDEX_NAME, TABLE_NAME, INDEX_COLUMNS)


def downgrade() -> None:
    """Drop the index. No column guard here, unlike upgrade().

    Removing an index needs nothing from the columns, and the index cannot
    exist if they do not. Raising would only block the rollback of a chain
    that is otherwise fine, on a database this revision never touched.
    """
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if INDEX_NAME in _index_names(insp, TABLE_NAME):
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)
