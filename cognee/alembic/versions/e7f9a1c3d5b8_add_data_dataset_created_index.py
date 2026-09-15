"""Add composite index on data for the dataset listing

Revision ID: e7f9a1c3d5b8
Revises: a7c2e9f4b8d1
Create Date: 2026-09-09

The HTTP listing uses (dataset_id, created_at DESC NULLS LAST, id).
Like c4e8a1f6b3d7, this startup migration stays inside Alembic's transaction:
CONCURRENTLY would release the version-row lock and allow concurrent workers
into the same index build. A regular build can block writes to a large data
table; deployments requiring an online build should pre-create this exact
index concurrently before starting the upgrade. The valid-index guard then
makes the startup revision a no-op. Failed earlier builds are repaired here.
Missing tables are skipped, matching the adapter's existence-guarded contract.
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import inspect, text

revision: str = "e7f9a1c3d5b8"
down_revision: str | None = "a7c2e9f4b8d1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "ix_data_dataset_created"


def upgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    if "data" not in inspector.get_table_names():
        return
    if conn.dialect.name == "postgresql":
        # Keep validity inspection and repair in the migration transaction.
        valid = conn.execute(
            text(
                "SELECT i.indisvalid FROM pg_index i "
                "JOIN pg_class c ON c.oid = i.indexrelid "
                "WHERE i.indrelid = 'data'::regclass AND c.relname = :name"
            ),
            {"name": INDEX_NAME},
        ).scalar_one_or_none()
        if valid is True:
            return
        if valid is False:
            op.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} ON data "
            "(dataset_id, created_at DESC NULLS LAST, id)"
        )
    elif INDEX_NAME not in {idx["name"] for idx in inspector.get_indexes("data")}:
        op.execute(f"CREATE INDEX {INDEX_NAME} ON data (dataset_id, created_at DESC, id)")


def downgrade() -> None:
    conn = op.get_bind()
    inspector = inspect(conn)
    if "data" not in inspector.get_table_names():
        return
    if INDEX_NAME in {idx["name"] for idx in inspector.get_indexes("data")}:
        op.drop_index(INDEX_NAME, table_name="data")
