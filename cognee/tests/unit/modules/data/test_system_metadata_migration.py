"""The system_metadata migration: column added, DLT stamps moved, users untouched.

Runs the real alembic migration module against a scratch SQLite database with
the pre-migration schema. Pins the contract of the split:

  - rows whose external_metadata carries a DLT stamp ("dlt"/"dlt_source") get
    the whole dict moved to system_metadata; their external_metadata is cleared
  - user-owned external_metadata — including one that happens to contain a
    "source" key with a non-DLT value — is never touched
  - re-running the migration is a no-op (idempotent)
"""

import importlib.util
import json
import pathlib

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "alembic"
    / "versions"
    / "f2b4c6d8e0a1_add_system_metadata_to_data.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("system_metadata_migration", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def old_schema_engine(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path}/old.db")
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE data (id CHAR(36) PRIMARY KEY, external_metadata JSON)"))
        rows = {
            "manifest": {"source": "dlt_source", "source_name": "people", "row_count": 3},
            "legacy": {"source": "dlt", "table": "people"},
            "user": {"notes": "mine", "source": "crm-export"},
            "empty": None,
        }
        for row_id, meta in rows.items():
            conn.execute(
                sa.text("INSERT INTO data (id, external_metadata) VALUES (:id, :meta)"),
                {"id": row_id, "meta": json.dumps(meta) if meta is not None else None},
            )
    return engine


def _run(engine, fn_name):
    module = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        module.op = Operations(ctx)
        getattr(module, fn_name)()


def _fetch(engine):
    with engine.connect() as conn:
        result = conn.execute(
            sa.text("SELECT id, external_metadata, system_metadata FROM data")
        ).fetchall()
    parsed = {}
    for row_id, ext, sysm in result:
        parsed[row_id] = (
            json.loads(ext) if isinstance(ext, str) else ext,
            json.loads(sysm) if isinstance(sysm, str) else sysm,
        )
    return parsed


def test_upgrade_moves_dlt_stamps_and_spares_users(old_schema_engine):
    _run(old_schema_engine, "upgrade")
    rows = _fetch(old_schema_engine)

    ext, sysm = rows["manifest"]
    assert sysm == {"source": "dlt_source", "source_name": "people", "row_count": 3}
    assert ext is None

    ext, sysm = rows["legacy"]
    assert sysm == {"source": "dlt", "table": "people"}
    assert ext is None

    # User metadata untouched — even with a (non-DLT) "source" key.
    ext, sysm = rows["user"]
    assert ext == {"notes": "mine", "source": "crm-export"}
    assert sysm is None

    ext, sysm = rows["empty"]
    assert ext is None and sysm is None


def test_upgrade_is_idempotent(old_schema_engine):
    _run(old_schema_engine, "upgrade")
    first = _fetch(old_schema_engine)
    _run(old_schema_engine, "upgrade")
    assert _fetch(old_schema_engine) == first
