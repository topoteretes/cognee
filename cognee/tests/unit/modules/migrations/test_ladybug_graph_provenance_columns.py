"""The ladybug_graph_provenance_columns migration upgrades legacy files.

Database files created before graph provenance (COG-5522) have Node/EDGE
tables without the four provenance columns; ``CREATE ... IF NOT EXISTS``
no-ops on them. An *empty* legacy graph then gets marked graph-provenance on
first write, after which every stamped write fails with "Cannot find property
source_run_refs" (caught live by the backwards-compatibility CI on a
v1.2.0-seeded database).

Runs the real migration against a real old-schema Ladybug file and then
performs the exact operation that failed in CI: mark the empty legacy graph
and write provenance-stamped nodes into it.
"""

import pathlib
from types import SimpleNamespace

import pytest

from cognee.modules.migrations.versions.ladybug_graph_provenance_columns import (
    _PROVENANCE_COLUMNS,
    downgrade,
    migrate,
)


def _create_legacy_database(db_path: pathlib.Path) -> None:
    """Create a database file with the pre-provenance (v1.2.0-era) schema."""
    import ladybug

    database = ladybug.Database(str(db_path))
    connection = ladybug.Connection(database)
    connection.execute("""
        CREATE NODE TABLE IF NOT EXISTS Node(
            id STRING PRIMARY KEY,
            name STRING,
            type STRING,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            properties STRING
        )
    """)
    connection.execute("""
        CREATE REL TABLE IF NOT EXISTS EDGE(
            FROM Node TO Node,
            relationship_name STRING,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            properties STRING
        )
    """)
    connection.close()
    database.close()


async def _table_columns(adapter, table: str) -> set[str]:
    rows = await adapter.query(f"CALL TABLE_INFO('{table}') RETURN *")
    return {row[1] for row in rows}


@pytest.fixture
def legacy_adapter(tmp_path):
    pytest.importorskip("ladybug")
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    db_path = tmp_path / "legacy_graph"
    _create_legacy_database(db_path)
    yield LadybugAdapter(str(db_path))


def _context(adapter) -> SimpleNamespace:
    return SimpleNamespace(graph_engine=adapter, vector_engine=None, dataset_id=None)


@pytest.mark.asyncio
async def test_migration_adds_missing_columns(legacy_adapter):
    for table in ("Node", "EDGE"):
        assert set(_PROVENANCE_COLUMNS) - await _table_columns(legacy_adapter, table)

    await migrate(_context(legacy_adapter))

    for table in ("Node", "EDGE"):
        missing = set(_PROVENANCE_COLUMNS) - await _table_columns(legacy_adapter, table)
        assert not missing, f"{table} still missing provenance columns: {missing}"

    # Idempotent: re-running must not raise or change anything.
    await migrate(_context(legacy_adapter))


@pytest.mark.asyncio
async def test_migrated_legacy_graph_accepts_stamped_writes(legacy_adapter):
    """The exact operation that failed in CI: marking the empty legacy graph
    graph-provenance and writing provenance-stamped nodes into it."""
    from uuid import UUID

    from cognee.infrastructure.databases.provenance.markers import (
        mark_graph_provenance_if_empty,
    )
    from cognee.infrastructure.databases.provenance.source_refs import make_source_ref_key
    from cognee.modules.engine.models import Entity

    await migrate(_context(legacy_adapter))

    marked = await mark_graph_provenance_if_empty(legacy_adapter)
    assert marked, "empty legacy graph must be markable after the migration"

    source_ref_key = make_source_ref_key(
        UUID("11111111-1111-5111-8111-111111111111"),
        UUID("22222222-2222-5222-8222-222222222222"),
    )
    await legacy_adapter.add_nodes(
        [Entity(name="compat-widget", description="legacy upgrade probe")],
        source_ref_key=source_ref_key,
        pipeline_run_id="33333333-3333-5333-8333-333333333333",
    )

    rows = await legacy_adapter.query("MATCH (n:Node) RETURN n.name, n.source_run_refs")
    assert len(rows) == 1
    assert rows[0][0] == "compat-widget"
    assert rows[0][1], "stamped write must populate source_run_refs"


@pytest.mark.asyncio
async def test_downgrade_removes_columns(legacy_adapter):
    await migrate(_context(legacy_adapter))
    await downgrade(_context(legacy_adapter))
    for table in ("Node", "EDGE"):
        assert set(_PROVENANCE_COLUMNS) - await _table_columns(legacy_adapter, table)
