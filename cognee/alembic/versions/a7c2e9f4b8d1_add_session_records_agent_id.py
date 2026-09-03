"""Add agent_id to session_records for agent-to-search attribution

Revision ID: a7c2e9f4b8d1
Revises: 1c22e6cb5aec
Create Date: 2026-09-03

SessionRecord gained a nullable, indexed ``agent_id`` column
(cognee/modules/session_lifecycle/models.py) so a session can be attributed
to the agent connection that opened it. Existing rows keep NULL; the
lifecycle code fills the column on the next touch of the session.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a7c2e9f4b8d1"
down_revision: Union[str, None] = "1c22e6cb5aec"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE_NAME = "session_records"
COLUMN_NAME = "agent_id"
INDEX_NAME = "ix_session_records_agent_id"


def _has_column(inspector, table: str, name: str) -> bool:
    return any(col["name"] == name for col in inspector.get_columns(table))


def _has_index(inspector, table: str, name: str) -> bool:
    return any(idx["name"] == name for idx in inspector.get_indexes(table))


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if not _has_column(insp, TABLE_NAME, COLUMN_NAME):
        op.add_column(TABLE_NAME, sa.Column(COLUMN_NAME, sa.String(), nullable=True))

    # Re-inspect: the column may have been added above, or by an out-of-band
    # create_all() that skipped the index.
    insp = sa.inspect(conn)
    if not _has_index(insp, TABLE_NAME, INDEX_NAME):
        op.create_index(INDEX_NAME, TABLE_NAME, [COLUMN_NAME], unique=False)


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if TABLE_NAME not in insp.get_table_names():
        return

    if _has_index(insp, TABLE_NAME, INDEX_NAME):
        op.drop_index(INDEX_NAME, table_name=TABLE_NAME)

    if _has_column(insp, TABLE_NAME, COLUMN_NAME):
        # batch mode: SQLite cannot DROP COLUMN in place.
        with op.batch_alter_table(TABLE_NAME) as batch_op:
            batch_op.drop_column(COLUMN_NAME)
