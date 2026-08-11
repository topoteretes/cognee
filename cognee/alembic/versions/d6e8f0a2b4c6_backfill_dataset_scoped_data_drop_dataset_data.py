"""backfill_dataset_scoped_data_drop_dataset_data

Completes the dataset-scoping of Data rows and retires the dataset_data
membership table:

1. Adds ``data.legacy_id`` (provenance for split rows).
2. Backfills every legacy row (``dataset_id IS NULL``) from its memberships:
   - exactly one membership → the row keeps its ORIGINAL id and is stamped
     with that dataset (user-held data_id mappings stay valid);
   - several memberships → the OLDEST membership keeps the original id; every
     other dataset gets a fresh row (new uuid4 id, all columns copied,
     ``legacy_id`` recording the original). The relational ledger
     (``nodes``) rows for those datasets are repointed to the new ids so
     ledger-driven deletion keeps working; their graph document nodes still
     carry the old id until the next full rebuild, which ``legacy_id``
     lets the delete path resolve.
   - zero memberships → unreachable orphans, left untouched.
3. Drops ``dataset_data``. Membership is now exactly ``Data.dataset_id``.

Revision ID: d6e8f0a2b4c6
Revises: c5d7e9f1a3b5
Create Date: 2026-08-11

"""

import uuid
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d6e8f0a2b4c6"
down_revision: Union[str, None] = "c5d7e9f1a3b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    tables = insp.get_table_names()

    if not _get_column(insp, "data", "legacy_id"):
        op.add_column("data", sa.Column("legacy_id", sa.UUID(), nullable=True))
        op.create_index("ix_data_legacy_id", "data", ["legacy_id"])

    if "dataset_data" not in tables:
        return

    # Lightweight TYPED table stubs (reflection mis-types some columns on
    # SQLite, e.g. NUMERIC-affinity reflections feeding Decimal processors).
    data_table = sa.table(
        "data",
        sa.column("id", sa.Uuid),
        sa.column("label", sa.String),
        sa.column("name", sa.String),
        sa.column("extension", sa.String),
        sa.column("mime_type", sa.String),
        sa.column("original_extension", sa.String),
        sa.column("original_mime_type", sa.String),
        sa.column("loader_engine", sa.String),
        sa.column("raw_data_location", sa.String),
        sa.column("original_data_location", sa.String),
        sa.column("owner_id", sa.Uuid),
        sa.column("tenant_id", sa.Uuid),
        sa.column("dataset_id", sa.Uuid),
        sa.column("legacy_id", sa.Uuid),
        sa.column("content_hash", sa.String),
        sa.column("raw_content_hash", sa.String),
        sa.column("external_metadata", sa.JSON),
        sa.column("node_set", sa.JSON),
        sa.column("pipeline_status", sa.JSON),
        sa.column("token_count", sa.Integer),
        sa.column("data_size", sa.Integer),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
        sa.column("last_accessed", sa.DateTime(timezone=True)),
        sa.column("importance_weight", sa.Float),
    )
    membership_table = sa.table(
        "dataset_data",
        sa.column("dataset_id", sa.Uuid),
        sa.column("data_id", sa.Uuid),
        sa.column("created_at", sa.DateTime(timezone=True)),
    )
    nodes_table = (
        sa.table(
            "nodes",
            sa.column("data_id", sa.Uuid),
            sa.column("dataset_id", sa.Uuid),
        )
        if "nodes" in tables
        else None
    )

    memberships = conn.execute(
        sa.select(
            membership_table.c.data_id,
            membership_table.c.dataset_id,
            membership_table.c.created_at,
        ).order_by(membership_table.c.created_at)
    ).fetchall()

    memberships_by_data: dict = {}
    for membership in memberships:
        memberships_by_data.setdefault(membership.data_id, []).append(membership)

    for data_id, data_memberships in memberships_by_data.items():
        row = (
            conn.execute(sa.select(data_table).where(data_table.c.id == data_id)).mappings().first()
        )
        if row is None or row["dataset_id"] is not None:
            # Missing row, or already dataset-scoped (new-scheme rows also
            # have a membership entry until this migration runs).
            continue

        # Oldest membership keeps the original id — sole-member rows (the
        # overwhelming majority) preserve their data_id exactly.
        keeper = data_memberships[0]
        conn.execute(
            sa.update(data_table)
            .where(data_table.c.id == data_id)
            .values(dataset_id=keeper.dataset_id)
        )

        for extra in data_memberships[1:]:
            new_id = uuid.uuid4()
            values = dict(row)
            values["id"] = new_id
            values["dataset_id"] = extra.dataset_id
            values["legacy_id"] = data_id
            conn.execute(sa.insert(data_table).values(**values))

            if nodes_table is not None:
                conn.execute(
                    sa.update(nodes_table)
                    .where(
                        nodes_table.c.data_id == data_id,
                        nodes_table.c.dataset_id == extra.dataset_id,
                    )
                    .values(data_id=new_id)
                )

    op.drop_table("dataset_data")


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "dataset_data" not in insp.get_table_names():
        op.create_table(
            "dataset_data",
            sa.Column("dataset_id", sa.UUID(), primary_key=True),
            sa.Column("data_id", sa.UUID(), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True)),
        )
        data_table = sa.table(
            "data",
            sa.column("id", sa.Uuid),
            sa.column("dataset_id", sa.Uuid),
            sa.column("created_at", sa.DateTime(timezone=True)),
        )
        membership_table = sa.table(
            "dataset_data",
            sa.column("dataset_id", sa.Uuid),
            sa.column("data_id", sa.Uuid),
            sa.column("created_at", sa.DateTime(timezone=True)),
        )
        rows = conn.execute(
            sa.select(data_table.c.id, data_table.c.dataset_id, data_table.c.created_at).where(
                data_table.c.dataset_id.is_not(None)
            )
        ).fetchall()
        for row in rows:
            conn.execute(
                sa.insert(membership_table).values(
                    dataset_id=row.dataset_id, data_id=row.id, created_at=row.created_at
                )
            )

    if _get_column(insp, "data", "legacy_id"):
        try:
            op.drop_index("ix_data_legacy_id", table_name="data")
        except Exception:
            pass
        op.drop_column("data", "legacy_id")
