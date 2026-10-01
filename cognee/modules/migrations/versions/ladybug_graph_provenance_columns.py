"""Backfill the graph-provenance columns onto pre-provenance Ladybug graphs.

COG-5522 added four STRING provenance columns (``source_ref_keys``,
``source_dataset_ids``, ``source_run_ids``, ``source_run_refs``) to the Node
and EDGE tables. Fresh databases get them from ``_ensure_schema``'s
``CREATE ... IF NOT EXISTS``; a database file created by a pre-provenance
release already has the tables, so that CREATE no-ops and the columns are
never added. An *empty* legacy graph then gets marked graph-provenance on
first write ("empty" was assumed to mean "fresh"), after which every stamped
write and provenance read fails with "Cannot find property source_run_refs"
— caught by the backwards-compatibility CI on a v1.2.0-seeded database.

This migration ALTER-adds exactly the missing columns. Like every migration it
is a frozen snapshot of the target schema — it deliberately does not import
the (mutable) adapter column list. No-op on every non-Ladybug graph backend.
"""

from __future__ import annotations

from cognee.modules.migrations.migration import MigrationContext

_PROVENANCE_COLUMNS = (
    "source_ref_keys",
    "source_dataset_ids",
    "source_run_ids",
    "source_run_refs",
)
_TABLES = ("Node", "EDGE")


def _is_ladybug_graph(graph_engine) -> bool:
    # Lazy, GUARDED import, mirroring postgres_graph_provenance_columns: the
    # registry imports every version module at startup, and this migration runs
    # for every graph backend. If the Ladybug adapter cannot be imported, the
    # engine cannot be a LadybugAdapter -> no-op.
    try:
        from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
    except ImportError:
        return False

    return isinstance(graph_engine, LadybugAdapter)


async def _existing_tables(graph_engine) -> set[str]:
    rows = await graph_engine.query("CALL SHOW_TABLES() RETURN *")
    # SHOW_TABLES rows: (id, name, type, database, comment) — name at index 1.
    return {row[1] for row in rows}


async def _table_columns(graph_engine, table: str) -> set[str]:
    rows = await graph_engine.query(f"CALL TABLE_INFO('{table}') RETURN *")
    # TABLE_INFO rows: (property id, name, type, default, pk) — name at index 1.
    return {row[1] for row in rows}


async def migrate(context: MigrationContext) -> None:
    graph_engine = context.graph_engine
    if not _is_ladybug_graph(graph_engine):
        return

    tables = await _existing_tables(graph_engine)
    for table in _TABLES:
        # A brand-new database may not have the tables when the chain runs;
        # _ensure_schema creates them with the columns already present.
        if table not in tables:
            continue
        existing = await _table_columns(graph_engine, table)
        for column in _PROVENANCE_COLUMNS:
            if column not in existing:
                await graph_engine.query(f"ALTER TABLE {table} ADD {column} STRING")


async def downgrade(context: MigrationContext) -> None:
    graph_engine = context.graph_engine
    if not _is_ladybug_graph(graph_engine):
        return

    tables = await _existing_tables(graph_engine)
    for table in _TABLES:
        if table not in tables:
            continue
        existing = await _table_columns(graph_engine, table)
        for column in _PROVENANCE_COLUMNS:
            if column in existing:
                await graph_engine.query(f"ALTER TABLE {table} DROP {column}")
