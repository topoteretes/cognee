"""Backfill the graph-provenance columns onto pre-provenance Postgres graphs.

The Postgres graph adapter (shipped since v1.0.x) stores nodes/edges in
``graph_node`` / ``graph_edge`` tables. COG-5522 added four ``varchar[]``
provenance columns (``source_ref_keys``, ``source_dataset_ids``,
``source_run_ids``, ``source_run_refs``) plus GIN indexes so delete/rollback can
filter by source ref, dataset id, or pipeline run id. Fresh databases get these
from ``create_all`` (see ``postgres/tables.py``); this migration adds them to a
graph_node/graph_edge that a pre-provenance release already created, since
``create_all(checkfirst=True)`` never ALTERs an existing table.

Only the Postgres graph backend is affected — for every other graph backend
(Ladybug/Kuzu, Neo4j, …) this is a no-op. The DDL is a frozen snapshot: like any
migration it is immutable history, so it does not import the (mutable) adapter
schema. Statements are idempotent (``IF NOT EXISTS``) and cheap on a fresh or
empty store; the constant ``'{}'`` default keeps ``ADD COLUMN`` metadata-only
(no table rewrite).
"""

from __future__ import annotations

from typing import Callable

from sqlalchemy import text

from cognee.modules.migrations.migration import MigrationContext

# The four provenance columns, and the three of them that carry a GIN membership
# index (source_run_refs is only ever read whole, never scanned by membership).
_PROVENANCE_COLUMNS = (
    "source_ref_keys",
    "source_dataset_ids",
    "source_run_ids",
    "source_run_refs",
)
_INDEXED_COLUMNS = ("source_ref_keys", "source_dataset_ids", "source_run_ids")
# (table name, index-name prefix) for the two graph tables.
_TABLES = (("graph_node", "node"), ("graph_edge", "edge"))


def _upgrade_statements(table: str, prefix: str) -> list[str]:
    return [
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} varchar[] NOT NULL DEFAULT '{{}}'"
        for column in _PROVENANCE_COLUMNS
    ] + [
        f"CREATE INDEX IF NOT EXISTS idx_{prefix}_{column} ON {table} USING gin ({column})"
        for column in _INDEXED_COLUMNS
    ]


def _downgrade_statements(table: str, prefix: str) -> list[str]:
    return [f"DROP INDEX IF EXISTS idx_{prefix}_{column}" for column in _INDEXED_COLUMNS] + [
        f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}" for column in _PROVENANCE_COLUMNS
    ]


def _is_postgres_graph(graph_engine) -> bool:
    # Lazy, GUARDED import. The registry imports every version module at startup,
    # and this migration runs for every graph backend — but the Postgres adapter
    # pulls in asyncpg / sqlalchemy-postgres, which ship only in the ``postgres``
    # extra. On a Ladybug/LanceDB-only install those are absent, so a bare import
    # would raise ModuleNotFoundError and crash the whole migration chain (blocking
    # writes) for a deployment that has no Postgres graph at all. If the adapter
    # cannot be imported, the engine cannot be a PostgresAdapter -> no-op.
    try:
        from cognee.infrastructure.databases.graph.postgres.adapter import PostgresAdapter
    except ImportError:
        return False

    return isinstance(graph_engine, PostgresAdapter)


async def _run(graph_engine, statements_for: Callable[[str, str], list[str]]) -> None:
    """Run each table's DDL, skipping tables that do not exist yet.

    A brand-new Postgres graph gets the columns from ``create_all`` and its
    graph_node/graph_edge may not exist when the chain runs on a fresh database —
    ``ALTER TABLE`` needs the table, so an absent one is skipped (create_all will
    build it with the columns already present).
    """
    async with graph_engine.engine.begin() as conn:
        for table, prefix in _TABLES:
            present = (await conn.execute(text(f"SELECT to_regclass('public.{table}')"))).scalar()
            if present is None:
                continue
            for statement in statements_for(table, prefix):
                await conn.execute(text(statement))


async def migrate(context: MigrationContext) -> None:
    if not _is_postgres_graph(context.graph_engine):
        return
    await _run(context.graph_engine, _upgrade_statements)


async def downgrade(context: MigrationContext) -> None:
    if not _is_postgres_graph(context.graph_engine):
        return
    await _run(context.graph_engine, _downgrade_statements)
