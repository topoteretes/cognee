"""Composite primary/foreign keys on cognee's two relational ingestion paths.

These are CONTRACT tests: each asserts the correct outcome and is expected to
FAIL until the underlying behaviour is fixed. Each test prints what the code
path actually produced (run with ``-s``) so a failure shows expected vs actual
side by side.

Fixture: ``loan_monthly_snapshot`` has a composite PK ``(loan_id, snapshot_date)``
with two rows for loan 1 (April, May); ``payment`` has a composite FK onto it,
one payment per loan, both for the May snapshot.

Legacy ``migrate_relational_database`` (reflects the user's source DB):
  1. every source row must become its own TableRow node (today: rows sharing
     the first key column collapse into one node, last row wins);
  2. a composite FK must become one edge per referencing row, to the row it
     actually references (today: the FK is split per column and the entry
     referencing the non-key column raises KeyError).

DLT path behind ``cognee.remember("<connection string>")`` (reflects dlt's
staging DB, which dlt writes without PK/FK constraints):
  3. every FK constraint of the source must reach the manifest and resolve to
     the referenced row (today: no FK survives staging, so no edge is drawn);
  4. a table with a primary key must be keyed by it (today: the PK is not
     reflected, the key falls back to `id` or the first column, and rows that
     share the fallback value shadow each other as FK targets).

Contract lines carry a `# CONTRACT` marker. Runs on the default local stack
(sqlite); no LLM, embeddings, or graph store.
"""

import ast
import importlib
import json
import pathlib
import re
import sqlite3

import pytest
import pytest_asyncio

DATASET = "composite_key_probe"
SNAPSHOT = "loan_monthly_snapshot"

# (loan_id, snapshot_date) of every snapshot row, and which one each payment references.
SNAPSHOT_ROWS = {(1, "2024-04-30"), (1, "2024-05-31"), (2, "2024-05-31")}
PAYMENT_TARGETS = {10: (1, "2024-05-31"), 11: (2, "2024-05-31")}

SOURCE_DDL = """
CREATE TABLE loan(id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE loan_monthly_snapshot(
  loan_id INTEGER NOT NULL, snapshot_date TEXT NOT NULL, balance REAL,
  PRIMARY KEY(loan_id, snapshot_date),
  FOREIGN KEY(loan_id) REFERENCES loan(id));
CREATE TABLE payment(
  id INTEGER PRIMARY KEY, loan_id INTEGER, snapshot_date TEXT, amount REAL,
  FOREIGN KEY(loan_id, snapshot_date) REFERENCES loan_monthly_snapshot(loan_id, snapshot_date));
INSERT INTO loan VALUES (1,'A'),(2,'B');
INSERT INTO loan_monthly_snapshot VALUES (1,'2024-04-30',100),(1,'2024-05-31',90),(2,'2024-05-31',50);
INSERT INTO payment VALUES (10,1,'2024-05-31',10),(11,2,'2024-05-31',5);
"""


def _clear_config_caches():
    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        ("cognee.infrastructure.databases.relational.config", "get_migration_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        (
            "cognee.infrastructure.databases.relational.get_migration_relational_engine",
            "get_migration_relational_engine",
        ),
        (
            "cognee.infrastructure.databases.relational.create_relational_engine",
            "create_relational_engine",
        ),
        ("cognee.tasks.ingestion.get_dlt_destination", "get_dlt_destination"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass


@pytest_asyncio.fixture
async def source_db(tmp_path, monkeypatch):
    """Path to a SQLite source DB; cognee + dlt state isolated under tmp_path."""
    pytest.importorskip("dlt")

    root = tmp_path
    for sub in ("system", "data", "db", "src", "dlt"):
        (root / sub).mkdir()
    src = root / "src" / "source.db"
    conn = sqlite3.connect(src)
    conn.executescript(SOURCE_DDL)
    conn.commit()
    conn.close()

    for key, value in {
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "TELEMETRY_DISABLED": "1",
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "DLT_DATA_DIR": str(root / "dlt"),
        # cognee's own relational store — dlt stages into a sibling DB in DB_PATH
        "DB_PROVIDER": "sqlite",
        "DB_PATH": str(root / "db"),
        "DB_NAME": "cognee_db",
        # legacy path reads the source through the migration engine
        "MIGRATION_DB_PROVIDER": "sqlite",
        "MIGRATION_DB_PATH": str(root / "src"),
        "MIGRATION_DB_NAME": "source.db",
    }.items():
        monkeypatch.setenv(key, value)
    _clear_config_caches()

    yield str(src)

    _clear_config_caches()


def _module(name: str):
    # cognee.tasks.ingestion re-exports the *functions* migrate_relational_database
    # and resolve_dlt_sources under their module names, so `import ... as m`
    # resolves to the function; import_module returns the module itself.
    return importlib.import_module(f"cognee.tasks.ingestion.{name}")


async def _source_schema(src: str) -> dict:
    """What SQLAlchemyAdapter.extract_schema() sees when pointed at the source."""
    from cognee.infrastructure.databases.relational.sqlalchemy.SqlAlchemyAdapter import (
        SQLAlchemyAdapter,
    )

    return await SQLAlchemyAdapter(f"sqlite+aiosqlite:///{src}").extract_schema()


def _source_pk(src: str, table: str) -> list[str]:
    conn = sqlite3.connect(src)
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info('{table}')") if c[5] > 0]
    conn.close()
    return cols


def _source_fk_constraints(src: str, table: str) -> int:
    """Number of FK constraints on a source table (PRAGMA rows grouped by constraint id)."""
    conn = sqlite3.connect(src)
    ids = {row[0] for row in conn.execute(f"PRAGMA foreign_key_list('{table}')")}
    conn.close()
    return len(ids)


# ---------------------------------------------------------------------------
# legacy: migrate_relational_database
# ---------------------------------------------------------------------------


def _row_key(table_row) -> tuple:
    """(loan_id, snapshot_date) of a snapshot TableRow, from its properties repr."""
    props = ast.literal_eval(table_row.properties)
    return (props["loan_id"], props["snapshot_date"])


def _table_of(node) -> str | None:
    """Table a legacy TableRow belongs to. TableRow has no `is_a` field (pydantic
    drops the kwarg the migration passes), so read it from the description the
    migration stamps: 'from the table with the name: "<table>"'."""
    match = re.search(r'table with the name: "([^"]+)"', getattr(node, "description", "") or "")
    return match.group(1) if match else None


def _snapshot_nodes(node_mapping) -> dict:
    """node_id -> TableRow for every snapshot row node in the mapping."""
    return {k: v for k, v in node_mapping.items() if _table_of(v) == SNAPSHOT}


@pytest.mark.asyncio
async def test_legacy_every_row_of_a_composite_pk_table_becomes_its_own_node(
    source_db, monkeypatch
):
    mrd = _module("migrate_relational_database")

    schema = await _source_schema(source_db)
    schema.pop("payment")  # isolate the PK contract from the composite-FK contract below

    created: list = []
    real_table_row = mrd.TableRow

    def recording_table_row(**kwargs):
        node = real_table_row(**kwargs)
        created.append(node)
        return node

    monkeypatch.setattr(mrd, "TableRow", recording_table_row)

    node_mapping, _ = await mrd.complete_database_ingestion(schema, migrate_column_data=False)

    built = [n for n in created if _table_of(n) == SNAPSHOT]
    kept = _snapshot_nodes(node_mapping)

    print(
        f"\n\nsource PK of {SNAPSHOT}: {_source_pk(source_db, SNAPSHOT)}"
        f"   extract_schema() primary_key: {schema[SNAPSHOT]['primary_key']!r}"
    )
    print(f"TableRow nodes built (source has {len(SNAPSHOT_ROWS)} rows):")
    print(f"  {'#':<3}{'node_id (name)':<28}{'uuid5(node_id)':<38}(loan_id, snapshot_date)")
    for i, node in enumerate(built, 1):
        print(f"  {i:<3}{node.name:<28}{node.id!s:<38}{_row_key(node)}")
    print(
        f"nodes that survived into node_mapping: {len(kept)}  -> {sorted(_row_key(n) for n in kept.values())}\n"
    )

    # CONTRACT: one node per source row — distinct ids, nothing overwritten.
    assert len({n.id for n in built}) == len(SNAPSHOT_ROWS), (
        f"{len(SNAPSHOT_ROWS)} source rows produced only {len({n.id for n in built})} distinct node ids: "
        f"{[n.name for n in built]}"
    )
    assert {_row_key(n) for n in kept.values()} == SNAPSHOT_ROWS, (
        f"rows missing from node_mapping: {SNAPSHOT_ROWS - {_row_key(n) for n in kept.values()}}"
    )


@pytest.mark.asyncio
async def test_legacy_composite_fk_becomes_one_edge_per_row_to_the_referenced_row(source_db):
    mrd = _module("migrate_relational_database")

    schema = await _source_schema(source_db)

    print(f"\n\nsource FK: payment(loan_id, snapshot_date) -> {SNAPSHOT}(loan_id, snapshot_date)")
    print("extract_schema()['payment']['foreign_keys']:")
    for fk in schema["payment"]["foreign_keys"]:
        print(f"  {fk}")

    # CONTRACT: a migration over a valid schema completes.
    try:
        node_mapping, edge_mapping = await mrd.complete_database_ingestion(
            schema, migrate_column_data=False
        )
    except KeyError as exc:
        pytest.fail(f"migration aborted with KeyError {exc} — composite FK was resolved per column")

    snapshots = _snapshot_nodes(node_mapping)
    snapshot_by_id = {n.id: _row_key(n) for n in snapshots.values()}
    payment_edges = {}
    for source_id, target_id, rel, _ in edge_mapping:
        if target_id in snapshot_by_id and source_id != target_id:
            payment_node = next((n for n in node_mapping.values() if n.id == source_id), None)
            if _table_of(payment_node) == "payment":
                pid = ast.literal_eval(payment_node.properties)["id"]
                payment_edges.setdefault(pid, []).append((snapshot_by_id[target_id], rel))

    print("payment -> snapshot edges produced:")
    for pid, targets in sorted(payment_edges.items()):
        print(f"  payment {pid}: {targets}")
    print(f"expected: {PAYMENT_TARGETS}\n")

    # CONTRACT: exactly one edge per payment row, aimed at the row it references.
    for pid, expected in PAYMENT_TARGETS.items():
        targets = [t for t, _ in payment_edges.get(pid, [])]
        assert targets == [expected], (
            f"payment {pid}: expected one edge to {expected}, got {targets}"
        )


# ---------------------------------------------------------------------------
# DLT: cognee.remember("<connection string>")
# ---------------------------------------------------------------------------


class _RecordingLogger:
    def __init__(self):
        self.records: list[tuple[str, str, tuple]] = []

    def _log(self, level, msg, *args, **kwargs):
        self.records.append((level, msg, args))

    def warning(self, msg, *args, **kwargs):
        self._log("warning", msg, *args)

    def info(self, msg, *args, **kwargs):
        self._log("info", msg, *args)

    def debug(self, *a, **k):
        pass


async def _dlt_manifest(
    src: str, monkeypatch
) -> tuple[dict, list, _RecordingLogger, _RecordingLogger]:
    """Run the DLT path up to the manifest (the step that mints node ids and
    resolves FKs). Returns (manifest, rows, ingest_log, resolve_log)."""
    from cognee.infrastructure.databases.relational import create_db_and_tables
    from cognee.modules.users.methods import get_default_user
    from cognee.tasks.ingestion.create_dlt_source import create_dlt_source_from_connection_string

    ids = _module("ingest_dlt_source")
    rds = _module("resolve_dlt_sources")
    ingest_log, resolve_log = _RecordingLogger(), _RecordingLogger()
    monkeypatch.setattr(ids, "logger", ingest_log)
    monkeypatch.setattr(rds, "logger", resolve_log)

    await create_db_and_tables()  # get_unique_data_id looks ids up in cognee's store
    user = await get_default_user()

    source = create_dlt_source_from_connection_string(f"sqlite:///{src}")
    rows = await ids.ingest_dlt_source(source, dataset_name=DATASET)
    item = await rds._build_source_manifest_item(rows, "source", DATASET, user)
    return json.loads(item.data), rows, ingest_log, resolve_log


def _manifest_row_data(row: dict) -> dict:
    """Parse the 'Row Data:' block of a manifest row's text back into a dict."""
    lines = row["text"].splitlines()
    start = lines.index("Row Data:") + 1
    data = {}
    for line in lines[start:]:
        key, _, value = line.strip().partition(": ")
        data[key] = value
    return data


def _manifest_key(row: dict) -> tuple:
    d = _manifest_row_data(row)
    return (int(d["loan_id"]), d["snapshot_date"])


@pytest.mark.asyncio
async def test_dlt_source_foreign_keys_become_references_to_the_referenced_rows(
    source_db, monkeypatch
):
    source_schema = await _source_schema(source_db)
    manifest, _rows, _, _ = await _dlt_manifest(source_db, monkeypatch)

    source_fk_count = sum(_source_fk_constraints(source_db, t) for t in source_schema)
    manifest_fk_count = sum(len(t["foreign_keys"]) for t in manifest["tables"].values())

    print(
        "\n\nforeign key constraints per table: source DB  vs  manifest['tables'][t]['foreign_keys']"
    )
    print(f"  {'table':<24}{'source':<10}manifest")
    for table in sorted(source_schema):
        print(
            f"  {table:<24}{_source_fk_constraints(source_db, table):<10}"
            f"{manifest['tables'][table]['foreign_keys']}"
        )

    node_key = {
        r["node_id"]: _manifest_key(r) for r in manifest["rows"] if r["table_name"] == SNAPSHOT
    }
    payment_refs = {}
    for r in manifest["rows"]:
        if r["table_name"] != "payment":
            continue
        pid = int(_manifest_row_data(r)["id"])
        payment_refs[pid] = [
            node_key.get(ref["target_node_id"], f"<{ref['target_table']}?>")
            for ref in r["fk_references"]
            if ref["target_table"] == SNAPSHOT
        ]
    print("payment rows' fk_references into snapshot rows:")
    for pid, targets in sorted(payment_refs.items()):
        print(f"  payment {pid}: {targets}")
    print(f"expected: {PAYMENT_TARGETS}\n")

    # CONTRACT: every FK constraint of the source is carried into the manifest ...
    assert manifest_fk_count == source_fk_count, (
        f"source has {source_fk_count} FK constraints (grouped, not per column), manifest carries {manifest_fk_count}"
    )
    # ... and each payment row references exactly the snapshot row it points at.
    for pid, expected in PAYMENT_TARGETS.items():
        assert payment_refs.get(pid) == [expected], (
            f"payment {pid}: expected one reference to {expected}, got {payment_refs.get(pid)}"
        )


@pytest.mark.asyncio
async def test_dlt_source_primary_key_is_used_so_rows_are_distinct_fk_targets(
    source_db, monkeypatch
):
    manifest, rows, ingest_log, resolve_log = await _dlt_manifest(source_db, monkeypatch)

    key_col = {r.table_name: r.primary_key_column for r in rows}
    print("\n\nkey column per table: real source PK  vs  what DLT keyed rows on")
    print(f"  {'table':<24}{'source PK constraint':<30}DltRowData.primary_key_column")
    for table in sorted(key_col):
        print(f"  {table:<24}{_source_pk(source_db, table)!s:<30}{key_col[table]!r}")

    snapshots = [r for r in manifest["rows"] if r["table_name"] == SNAPSHOT]
    print(f"\nmanifest rows for {SNAPSHOT}:")
    for r in sorted(snapshots, key=_manifest_key):
        print(f"  {r['node_id']}  primary_key_value={r['primary_key_value']!r}  {_manifest_key(r)}")
    fallback = [a for lvl, m, a in ingest_log.records if "no primary key detected" in m]
    shadow = [a for lvl, m, a in resolve_log.records if "Duplicate primary key" in m]
    print(f"\n'no primary key detected' fallbacks: {fallback}")
    print(f"'Duplicate primary key during FK resolution' warnings: {shadow}\n")

    # CONTRACT: a table that declares a PK is keyed by it — no guessing ...
    assert not fallback, f"PK not reflected; fell back to first column for: {fallback}"
    # ... and the key distinguishes every row, so no row shadows another as an FK target.
    assert len({r["primary_key_value"] for r in snapshots}) == len(SNAPSHOT_ROWS), (
        f"key values not unique per row: {[r['primary_key_value'] for r in snapshots]}"
    )
    assert not shadow, f"rows shadow each other in fk_lookup: {shadow}"
