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

Built to execute correctly at production scale: sole-membership stamping (the
overwhelming majority of rows) is ONE set-based UPDATE — no per-row round
trips, no Python-side materialization; only the shared groups (rare) are
walked in Python, and fork rows are copied server-side (INSERT ... FROM
SELECT), so column values never round-trip through Python typing. IN-lists
are chunked to stay under SQLite's bound-parameter limit.

Atomicity: on PostgreSQL alembic wraps the migration in a transaction; on
SQLite it does NOT (non-transactional-DDL mode makes ``begin_transaction`` a
no-op, so every statement would autocommit — one fsync per statement, and a
mid-run crash would persist partial work). The backfill therefore opens its
own transaction when the context did not provide one: a crash rolls back
cleanly on both dialects and the migration simply reruns. Pre-flight and
completion counts go to the log.

NOT an online migration: old-version replicas write shared rows and read
``dataset_data``, both of which this migration retires — stop all replicas,
upgrade, then start. A concurrent second run self-cancels (both transactions
contain the ``dataset_data`` drop; the loser rolls back whole).

Revision ID: d6e8f0a2b4c6
Revises: c5d7e9f1a3b5
Create Date: 2026-08-11

"""

import logging
import uuid
from contextlib import nullcontext
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

logger = logging.getLogger(__name__)

# SQLite's default bound-parameter limit is 999; chunk IN-lists well under it.
_IN_CLAUSE_CHUNK = 900


def _chunks(values, size=_IN_CLAUSE_CHUNK):
    for start in range(0, len(values), size):
        yield values[start : start + size]


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

    # On SQLite alembic's migration context is non-transactional, so every
    # statement below would autocommit (fsync each) and a crash would persist
    # partial work. Open our own transaction when none is active; on
    # PostgreSQL the context's transaction is already in place.
    transaction_scope = conn.begin() if not conn.in_transaction() else nullcontext()
    with transaction_scope:
        _backfill(conn, data_table, membership_table, nodes_table)


def _backfill(conn, data_table, membership_table, nodes_table) -> None:
    # The legacy membership PK is (dataset_id, data_id) — the shared-group
    # probes below go BY data_id, which that index cannot serve. Build a
    # temporary index for the migration's lifetime (dropped with the table).
    op.create_index(
        "ix_backfill_dataset_data_data_id", "dataset_data", ["data_id"], if_not_exists=True
    )

    # ── Pre-flight: size the work before doing any of it ──────────────────
    legacy_row_count = conn.execute(
        sa.select(sa.func.count()).select_from(data_table).where(data_table.c.dataset_id.is_(None))
    ).scalar()
    shared_ids = [
        row[0]
        for row in conn.execute(
            sa.select(membership_table.c.data_id)
            .group_by(membership_table.c.data_id)
            .having(sa.func.count() > 1)
        )
    ]
    logger.info(
        "dataset-scoping backfill: %s legacy row(s) to stamp, %s shared row(s) to split",
        legacy_row_count,
        len(shared_ids),
    )

    # ── Sole-membership rows: ONE set-based UPDATE ... FROM stamps them ────
    # The sole memberships are materialized once (a semi-join against the
    # count-1 groups — no UUID aggregation, which Postgres lacks) and
    # joined by primary key. Deliberately NOT a correlated subquery: SQLite
    # gives the model's UUID-typed columns NUMERIC affinity, and correlated
    # per-row probes against the TEXT-typed membership column degrade to
    # full scans with a string-to-float conversion attempt per comparison
    # (sqlite3AtoF dominating the profile). The row keeps its original id;
    # only dataset_id moves. Already-scoped rows (which also carry a
    # membership entry until this runs) are excluded by the predicate.
    sole_ids = (
        sa.select(membership_table.c.data_id)
        .group_by(membership_table.c.data_id)
        .having(sa.func.count() == 1)
    )
    sole_memberships = (
        sa.select(
            membership_table.c.data_id.label("data_id"),
            membership_table.c.dataset_id.label("dataset_id"),
        )
        .where(membership_table.c.data_id.in_(sole_ids))
        .subquery("sole_memberships")
    )
    stamped = conn.execute(
        sa.update(data_table)
        .where(
            data_table.c.id == sole_memberships.c.data_id,
            data_table.c.dataset_id.is_(None),
        )
        .values(dataset_id=sole_memberships.c.dataset_id)
    )
    logger.info("dataset-scoping backfill: stamped %s sole-membership row(s)", stamped.rowcount)

    # ── Shared rows (rare): oldest membership keeps the id, others fork ────
    # Only this small set is walked in Python; the fork copy itself is a
    # server-side INSERT ... FROM SELECT, so the row's columns never
    # round-trip through Python typing.
    forks_created = 0
    if shared_ids:
        shared_row_datasets: dict = {}
        memberships_by_data: dict = {}
        for chunk in _chunks(shared_ids):
            for row in conn.execute(
                sa.select(data_table.c.id, data_table.c.dataset_id).where(
                    data_table.c.id.in_(chunk)
                )
            ):
                shared_row_datasets[row.id] = row.dataset_id
            for membership in conn.execute(
                sa.select(membership_table.c.data_id, membership_table.c.dataset_id)
                .where(membership_table.c.data_id.in_(chunk))
                .order_by(membership_table.c.created_at)
            ):
                memberships_by_data.setdefault(membership.data_id, []).append(membership.dataset_id)

        column_names = [column.name for column in data_table.columns]
        for data_id, dataset_ids in memberships_by_data.items():
            if data_id not in shared_row_datasets or shared_row_datasets[data_id] is not None:
                # Missing row, or already dataset-scoped. Skipping BEFORE the
                # ledger repoint matters: a membership whose data row is gone
                # must not repoint ledger rows at a fork that is never created.
                continue

            keeper_dataset = dataset_ids[0]
            conn.execute(
                sa.update(data_table)
                .where(data_table.c.id == data_id, data_table.c.dataset_id.is_(None))
                .values(dataset_id=keeper_dataset)
            )

            for fork_dataset in dataset_ids[1:]:
                new_id = uuid.uuid4()
                fork_columns = [
                    sa.literal(new_id, sa.Uuid())
                    if name == "id"
                    else sa.literal(fork_dataset, sa.Uuid())
                    if name == "dataset_id"
                    else data_table.c.id
                    if name == "legacy_id"
                    else data_table.c[name]
                    for name in column_names
                ]
                conn.execute(
                    sa.insert(data_table).from_select(
                        column_names,
                        sa.select(*fork_columns).where(data_table.c.id == data_id),
                    )
                )
                forks_created += 1

                if nodes_table is not None:
                    conn.execute(
                        sa.update(nodes_table)
                        .where(
                            nodes_table.c.data_id == data_id,
                            nodes_table.c.dataset_id == fork_dataset,
                        )
                        .values(data_id=new_id)
                    )

    logger.info(
        "dataset-scoping backfill: split %s shared row(s) into %s fork(s)",
        len(shared_ids),
        forks_created,
    )

    op.drop_table("dataset_data")


def downgrade() -> None:
    """Restore the old schema shape: recreate ``dataset_data`` (one membership
    per scoped row — data-preserving, never id-merging) and drop ``legacy_id``.

    ROLLBACK ORDER: run the ``rekey_fork_document_ids`` chain migration's
    downgrade BEFORE this one. Its reverse map is built from ``legacy_id``;
    dropping the column first leaves fork graphs stranded on canonical ids
    with no way back. Dropping ``legacy_id`` also loses fork lineage
    permanently — the old schema has nowhere to keep it, so a later
    re-upgrade re-splits shared rows with fresh ids.
    """
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
        # One membership per scoped row, written server-side in one statement.
        conn.execute(
            sa.insert(membership_table).from_select(
                ["dataset_id", "data_id", "created_at"],
                sa.select(data_table.c.dataset_id, data_table.c.id, data_table.c.created_at).where(
                    data_table.c.dataset_id.is_not(None)
                ),
            )
        )

    if _get_column(insp, "data", "legacy_id"):
        try:
            op.drop_index("ix_data_legacy_id", table_name="data")
        except Exception:
            pass
        op.drop_column("data", "legacy_id")
