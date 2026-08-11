"""The dataset-scoping backfill migration, tested against a simulated legacy DB.

Builds a scratch sqlite database in the PRE-migration shape — current model
tables plus a hand-created ``dataset_data`` membership table and NULL-scoped
legacy rows — then runs the real migration's ``upgrade()`` through an alembic
``MigrationContext`` and asserts the contract:

  - a sole-membership legacy row KEEPS its original data_id and is stamped
    with its dataset (user-held id mappings stay valid);
  - a multi-membership legacy row keeps the original id for its OLDEST
    membership; every other dataset gets a fresh row with a new id, all
    columns copied, ``legacy_id`` recording the original, and its
    relational ledger (``nodes``) rows repointed to the new id;
  - already-scoped rows are untouched;
  - ``dataset_data`` is dropped;
  - downgrade recreates ``dataset_data`` with one membership per scoped row
    and drops ``legacy_id`` — data-preserving, never id-merging.
"""

import importlib
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

migration = importlib.import_module(
    "cognee.alembic.versions.d6e8f0a2b4c6_backfill_dataset_scoped_data_drop_dataset_data"
)


def _make_engine():
    db_path = Path(tempfile.mkdtemp(prefix="cognee_backfill_test_")) / "test.db"
    return sa.create_engine(f"sqlite:///{db_path}")


def _create_schema(engine):
    # Register the current models, then add the dropped membership table by hand.
    from cognee.infrastructure.databases.relational import Base
    from cognee.modules.data.models import Data, Dataset  # noqa: F401
    from cognee.modules.graph.models import Node  # noqa: F401

    Base.metadata.create_all(engine, tables=[Data.__table__, Dataset.__table__, Node.__table__])
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE dataset_data ("
                "dataset_id CHAR(32) NOT NULL, data_id CHAR(32) NOT NULL, "
                "created_at DATETIME, PRIMARY KEY (dataset_id, data_id))"
            )
        )


def test_backfill_preserves_sole_ids_and_splits_shared_rows():
    engine = _make_engine()
    _create_schema(engine)

    from cognee.modules.data.models import Data
    from cognee.modules.graph.models import Node

    dataset_a, dataset_b = uuid.uuid4(), uuid.uuid4()
    sole_id, shared_id, scoped_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    now = datetime.now(timezone.utc)

    data_table = Data.__table__
    nodes_table = Node.__table__

    with engine.begin() as conn:

        def add_row(row_id, dataset_id=None, name="doc"):
            conn.execute(
                sa.insert(data_table).values(
                    id=row_id,
                    dataset_id=dataset_id,
                    name=name,
                    content_hash="hash-" + name,
                    raw_data_location=f"file:///tmp/{name}.txt",
                    owner_id=uuid.uuid4(),
                    pipeline_status={},
                    token_count=-1,
                )
            )

        add_row(sole_id, name="sole")
        add_row(shared_id, name="shared")
        add_row(scoped_id, dataset_id=dataset_a, name="scoped")

        def add_membership(dataset_id, data_id, created_at):
            conn.execute(
                sa.text(
                    "INSERT INTO dataset_data (dataset_id, data_id, created_at) "
                    "VALUES (:ds, :da, :ts)"
                ),
                {"ds": dataset_id.hex, "da": data_id.hex, "ts": created_at},
            )

        add_membership(dataset_a, sole_id, now)
        add_membership(dataset_a, shared_id, now)  # oldest -> keeper
        add_membership(dataset_b, shared_id, now + timedelta(minutes=1))
        add_membership(dataset_a, scoped_id, now)

        # Ledger rows for the shared document in both datasets.
        for ds in (dataset_a, dataset_b):
            conn.execute(
                sa.insert(nodes_table).values(
                    id=uuid.uuid4(),
                    slug=uuid.uuid4(),
                    user_id=uuid.uuid4(),
                    data_id=shared_id,
                    dataset_id=ds,
                    type="DocumentChunk",
                    indexed_fields=[],
                )
            )

    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        with context.begin_transaction():
            with Operations.context(context):
                migration.upgrade()
        conn.commit()

    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                data_table.c.id,
                data_table.c.dataset_id,
                data_table.c.legacy_id,
                data_table.c.name,
            )
        ).fetchall()
        by_id = {row.id: row for row in rows}

        # Sole-membership: original id kept, dataset stamped.
        assert by_id[sole_id].dataset_id == dataset_a
        assert by_id[sole_id].legacy_id is None

        # Already-scoped: untouched.
        assert by_id[scoped_id].dataset_id == dataset_a
        assert by_id[scoped_id].legacy_id is None

        # Shared: keeper (oldest membership, dataset_a) retains the id...
        assert by_id[shared_id].dataset_id == dataset_a
        # ...and dataset_b got a fresh row copied from it.
        split_rows = [row for row in rows if row.legacy_id == shared_id]
        assert len(split_rows) == 1
        split_row = split_rows[0]
        assert split_row.dataset_id == dataset_b
        assert split_row.id != shared_id
        assert split_row.name == "shared"

        # Ledger repoint: dataset_b's nodes follow the new id; dataset_a's stay.
        ledger = conn.execute(sa.select(nodes_table.c.dataset_id, nodes_table.c.data_id)).fetchall()
        ledger_by_dataset = {row.dataset_id: row.data_id for row in ledger}
        assert ledger_by_dataset[dataset_a] == shared_id
        assert ledger_by_dataset[dataset_b] == split_row.id

        # Membership table is gone.
        assert "dataset_data" not in sa.inspect(conn).get_table_names()


def test_backfill_downgrade_recreates_memberships_without_data_loss():
    """Downgrade restores the old schema shape with zero row loss.

    ``dataset_data`` is recreated with one membership per scoped row (fork
    rows stay separate rows — merging them back could destroy post-upgrade
    divergence, so the downgrade is deliberately data-preserving, not
    id-merging) and the ``legacy_id`` column is dropped. Pinned consequence:
    dropping the column loses fork lineage — a later re-upgrade cannot
    restore ``legacy_id`` values, so pre-fork ids stop resolving for fork
    rows. That is inherent to the old schema having nowhere to keep lineage.
    """
    engine = _make_engine()
    _create_schema(engine)

    from cognee.modules.data.models import Data

    dataset_a, dataset_b = uuid.uuid4(), uuid.uuid4()
    sole_id, shared_id = uuid.uuid4(), uuid.uuid4()
    now = datetime.now(timezone.utc)

    data_table = Data.__table__

    with engine.begin() as conn:

        def add_row(row_id, dataset_id=None, name="doc"):
            conn.execute(
                sa.insert(data_table).values(
                    id=row_id,
                    dataset_id=dataset_id,
                    name=name,
                    content_hash="hash-" + name,
                    raw_data_location=f"file:///tmp/{name}.txt",
                    owner_id=uuid.uuid4(),
                    pipeline_status={},
                    token_count=-1,
                )
            )

        add_row(sole_id, name="sole")
        add_row(shared_id, name="shared")

        conn.execute(
            sa.text(
                "INSERT INTO dataset_data (dataset_id, data_id, created_at) VALUES (:ds, :da, :ts)"
            ),
            [
                {"ds": dataset_a.hex, "da": sole_id.hex, "ts": now},
                {"ds": dataset_a.hex, "da": shared_id.hex, "ts": now},
                {"ds": dataset_b.hex, "da": shared_id.hex, "ts": now + timedelta(minutes=1)},
            ],
        )

    def run(operation):
        with engine.connect() as conn:
            context = MigrationContext.configure(conn)
            with context.begin_transaction():
                with Operations.context(context):
                    operation()
            conn.commit()

    run(migration.upgrade)

    with engine.connect() as conn:
        upgraded = conn.execute(
            sa.select(data_table.c.id, data_table.c.dataset_id, data_table.c.legacy_id)
        ).fetchall()
    assert len(upgraded) == 3, "upgrade split the shared row into two scoped rows"
    split_row = next(row for row in upgraded if row.legacy_id == shared_id)

    run(migration.downgrade)

    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        assert "dataset_data" in inspector.get_table_names(), "membership table recreated"
        assert "legacy_id" not in {c["name"] for c in inspector.get_columns("data")}, (
            "legacy_id column dropped"
        )

        rows = conn.execute(sa.select(data_table.c.id, data_table.c.dataset_id)).fetchall()
        assert {row.id for row in rows} == {sole_id, shared_id, split_row.id}, (
            "no rows lost or merged on downgrade"
        )

        memberships = {
            (row.dataset_id, row.data_id)
            for row in conn.execute(sa.text("SELECT dataset_id, data_id FROM dataset_data"))
        }
        assert memberships == {
            (dataset_a.hex, sole_id.hex),
            (dataset_a.hex, shared_id.hex),
            (dataset_b.hex, split_row.id.hex),
        }, "one membership per scoped row, fork membership points at the fork row"
