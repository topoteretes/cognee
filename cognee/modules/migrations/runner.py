"""Startup runner that applies pending graph/vector migrations to every database.

- Access control ON  -> one ``dataset_database`` row per dataset carries that
  database pair's revisions; each row's databases are resolved through the
  per-dataset context and migrated independently.
- Access control OFF -> a single global graph/vector pair backs every dataset,
  so revisions live in the standalone single-row ``global_database_version``
  table (one database, one row — per-dataset tracking is meaningless here).

In both modes the runner walks the stored revision forward to head (see
``migration.py``) and records the new head revision (plus, as an audit value,
the current Cognee version).

Concurrency: the migrate-then-stamp sequence runs under a cross-process mutex.
On Postgres this is a session-scoped advisory lock held on a dedicated
connection with NO open transaction, so a long migration neither blocks row
access nor trips idle-in-transaction timeouts; the stored revisions are
re-read after acquiring the lock, so the loser of a startup race sees the
winner's stamp and skips. On SQLite there is NO cross-process lock — running
multiple workers against one SQLite metadata store during a migration window
is not supported (the post-lock re-read narrows the race but cannot close it).
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from cognee.version import get_cognee_version
from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.users.models import DatasetDatabase

from cognee.modules.migrations.migration import (
    Migration,
    MigrationContext,
    migrations_to_downgrade,
    pending_migrations,
)
from cognee.modules.migrations.models import GLOBAL_DATABASE_VERSION_ROW_ID, GlobalDatabaseVersion
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

logger = logging.getLogger(__name__)

# Advisory-lock key for the global database pair (any stable bigint works).
_GLOBAL_MIGRATION_LOCK_KEY = 0x636F676E6565_01  # "cognee" + 01


def _advisory_key(dataset_id: UUID) -> int:
    """Stable 64-bit advisory-lock key for one dataset's migration mutex."""
    return int.from_bytes(dataset_id.bytes[:8], "big", signed=True)


@asynccontextmanager
async def _migration_lock(db_engine, key: int):
    """Cross-process mutex around one database's migrate-then-stamp sequence.

    Postgres: session-scoped ``pg_advisory_lock`` on a dedicated connection.
    The transaction is committed immediately after acquiring (advisory session
    locks survive commit), so nothing relational stays locked or open while
    the migration runs. Other dialects (SQLite): no cross-process primitive
    exists — yields without locking; see the module docstring.
    """
    engine = db_engine.engine
    if engine.dialect.name != "postgresql":
        yield
        return

    async with engine.connect() as connection:
        await connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        await connection.commit()
        try:
            yield
        finally:
            await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
            await connection.commit()


async def _apply(
    context: MigrationContext, migrations: list[Migration], stored_revision: str | None
) -> tuple[list[str], str | None]:
    """Run every pending migration in order.

    Returns the applied migration slugs and the resulting revision (the chain
    head if anything ran, otherwise the unchanged stored revision).
    """
    pending = pending_migrations(migrations, stored_revision)
    for migration in pending:
        logger.info(
            "Applying migration '%s' (cognee %s).", migration.slug, migration.cognee_version
        )
        await migration.up(context)

    new_revision = pending[-1].revision if pending else stored_revision
    return [migration.slug for migration in pending], new_revision


def _nothing_pending(graph_revision: str | None, vector_revision: str | None) -> bool:
    """True when both revision chains are already satisfied by the given snapshot."""
    return not pending_migrations(GRAPH_MIGRATIONS, graph_revision) and not pending_migrations(
        VECTOR_MIGRATIONS, vector_revision
    )


async def pending_migration_dataset_ids() -> list[UUID]:
    """Dataset ids whose databases have unapplied migrations.

    Per-dataset (access control on) mode only; in global mode use
    :func:`global_migrations_pending`. Cheap: reads the rows and compares
    revisions in memory, no engines are opened.
    """
    rows = await get_dataset_databases()
    return [
        row.dataset_id
        for row in rows
        if not _nothing_pending(row.graph_migration_revision, row.vector_migration_revision)
    ]


async def global_migrations_pending() -> bool:
    """True when the global database pair (access control off) has unapplied migrations."""
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        row = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
    if row is None:
        return True
    return not _nothing_pending(
        row.global_graph_migration_revision, row.global_vector_migration_revision
    )


async def _record_deployment_version(current_version: str) -> GlobalDatabaseVersion:
    """Upsert the single ``global_database_version`` row's ``cognee_version``.

    Runs on every startup in BOTH access-control modes, so this row is the one
    place to read which Cognee release last ran. The global revision columns
    are left untouched here (NULL on creation): in per-dataset mode they stay
    NULL forever, and in global mode NULL means "run every migration" — correct
    for an upgrade, and a free no-op chain run followed by a head stamp for a
    fresh deployment (every migration is idempotent and cheap on empty stores).
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        if record is not None:
            if record.cognee_version != current_version:
                record.cognee_version = current_version
                await session.commit()
            return record

    async with db_engine.get_async_session() as session:
        record = GlobalDatabaseVersion(
            id=GLOBAL_DATABASE_VERSION_ROW_ID,
            cognee_version=current_version,
        )
        try:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record
        except IntegrityError:
            # Concurrent startup (e.g. multiple workers) already created it.
            await session.rollback()
            existing = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if existing is None:
                raise
            return existing


async def _run_global_migrations(current_version: str) -> list[dict]:
    """Migrate the single global graph/vector pair (access control disabled).

    Same lock + re-check + migrate + stamp sequence as the per-dataset path,
    against the ``global_database_version`` row. The migration context carries
    ``dataset_id=None``, so ledger updates apply unscoped — correct here, since
    the one global graph backs every dataset's ledger rows.
    """
    row = await _record_deployment_version(current_version)
    if _nothing_pending(row.global_graph_migration_revision, row.global_vector_migration_revision):
        return [
            {
                "database": "global",
                "graph_migrations_applied": [],
                "vector_migrations_applied": [],
            }
        ]

    db_engine = get_relational_engine()
    try:
        async with _migration_lock(db_engine, _GLOBAL_MIGRATION_LOCK_KEY):
            # Re-read under the lock: a concurrent worker may have finished.
            async with db_engine.get_async_session() as session:
                record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is None or _nothing_pending(
                record.global_graph_migration_revision, record.global_vector_migration_revision
            ):
                return []
            graph_stored = record.global_graph_migration_revision
            vector_stored = record.global_vector_migration_revision

            # No context override: without access control, get_graph_engine /
            # get_vector_engine resolve the global databases directly.
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=None,
            )
            graph_applied, graph_revision = await _apply(
                migration_context, GRAPH_MIGRATIONS, graph_stored
            )
            vector_applied, vector_revision = await _apply(
                migration_context, VECTOR_MIGRATIONS, vector_stored
            )

            async with db_engine.get_async_session() as session:
                record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
                if record is not None:
                    record.global_graph_migration_revision = graph_revision
                    record.global_vector_migration_revision = vector_revision
                    await session.commit()
    except Exception:
        logger.exception("Database migrations failed for the global databases.")
        return [{"database": "global", "result": "failed"}]

    if graph_applied or vector_applied:
        logger.info("Migrated global databases: graph=%s vector=%s.", graph_applied, vector_applied)
    return [
        {
            "database": "global",
            "graph_migrations_applied": graph_applied,
            "vector_migrations_applied": vector_applied,
        }
    ]


async def _migrate_dataset(db_engine, row, current_version: str) -> Optional[dict]:
    """Run pending migrations for one dataset's database pair, under its lock.

    Relational transactions stay SHORT: one read after acquiring the lock, the
    migration itself runs with nothing relational open, one write to stamp.
    Returns a summary dict, or ``None`` when there was nothing to do (or the
    row vanished — dataset deleted concurrently).
    """
    async with _migration_lock(db_engine, _advisory_key(row.dataset_id)):
        # Re-read under the lock: a concurrent worker may have finished.
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                return None
            graph_stored = record.graph_migration_revision
            vector_stored = record.vector_migration_revision
        if _nothing_pending(graph_stored, vector_stored):
            return None

        # Resolve this dataset's graph/vector databases through the
        # per-dataset context — the same way every other operation does.
        async with set_database_global_context_variables(row.dataset_id, row.owner_id):
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=row.dataset_id,
            )
            graph_applied, graph_revision = await _apply(
                migration_context, GRAPH_MIGRATIONS, graph_stored
            )
            vector_applied, vector_revision = await _apply(
                migration_context, VECTOR_MIGRATIONS, vector_stored
            )

        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is not None:
                record.graph_migration_revision = graph_revision
                record.vector_migration_revision = vector_revision
                # Audit only: the release that last migrated this dataset.
                record.cognee_version = current_version
                await session.commit()

    return {
        "dataset_id": str(row.dataset_id),
        "graph_migrations_applied": graph_applied,
        "vector_migrations_applied": vector_applied,
    }


async def run_database_migrations() -> list[dict]:
    """Apply pending graph and vector migrations to every Cognee database.

    Failures for one database are logged and skipped so the remaining databases
    are still migrated. Returns a per-database summary.
    """
    current_version = get_cognee_version()

    if not backend_access_control_enabled():
        return await _run_global_migrations(current_version)

    # Record the deployment-wide version even in per-dataset mode (the global
    # revision columns stay NULL — per-dataset revisions live on each row).
    await _record_deployment_version(current_version)

    rows = await get_dataset_databases()
    db_engine = get_relational_engine()

    summaries: list[dict] = []

    for row in rows:
        # Fast path: nothing pending per this row's snapshot — skip without
        # locking, opening the dataset's databases, or writing anything.
        if _nothing_pending(row.graph_migration_revision, row.vector_migration_revision):
            summaries.append(
                {
                    "dataset_id": str(row.dataset_id),
                    "graph_migrations_applied": [],
                    "vector_migrations_applied": [],
                }
            )
            continue

        try:
            summary = await _migrate_dataset(db_engine, row, current_version)
        except Exception:
            logger.exception(
                "Database migrations failed for dataset '%s'; continuing with remaining datasets.",
                row.dataset_id,
            )
            summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue

        if summary is None:
            continue
        summaries.append(summary)
        if summary["graph_migrations_applied"] or summary["vector_migrations_applied"]:
            logger.info(
                "Migrated dataset '%s': graph=%s vector=%s.",
                row.dataset_id,
                summary["graph_migrations_applied"],
                summary["vector_migrations_applied"],
            )

    return summaries


async def _revert(
    context: MigrationContext,
    migrations: list[Migration],
    stored_revision: Optional[str],
    target_revision: Optional[str],
) -> list[str]:
    """Run the down() of every migration between stored and target, newest first."""
    to_revert = migrations_to_downgrade(migrations, stored_revision, target_revision)
    for migration in to_revert:
        logger.info("Reverting migration '%s'.", migration.slug)
        await migration.down(context)
    return [migration.slug for migration in to_revert]


async def _downgrade_dataset(
    db_engine, row, graph_target: Optional[str], vector_target: Optional[str]
) -> Optional[dict]:
    """Revert migrations for one dataset's database pair, under its lock."""
    async with _migration_lock(db_engine, _advisory_key(row.dataset_id)):
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                return None
            graph_stored = record.graph_migration_revision
            vector_stored = record.vector_migration_revision

        async with set_database_global_context_variables(row.dataset_id, row.owner_id):
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=row.dataset_id,
            )
            # Reverse of the upgrade order: vector chain first, then graph.
            vector_reverted = await _revert(
                migration_context, VECTOR_MIGRATIONS, vector_stored, vector_target
            )
            graph_reverted = await _revert(
                migration_context, GRAPH_MIGRATIONS, graph_stored, graph_target
            )

        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is not None:
                if graph_reverted:
                    record.graph_migration_revision = graph_target
                if vector_reverted:
                    record.vector_migration_revision = vector_target
                await session.commit()

    return {
        "dataset_id": str(row.dataset_id),
        "graph_migrations_reverted": graph_reverted,
        "vector_migrations_reverted": vector_reverted,
    }


async def downgrade_database_migrations(
    graph_target_revision: Optional[str] = None,
    vector_target_revision: Optional[str] = None,
) -> list[dict]:
    """Revert graph/vector migrations back to the given target revisions.

    EXPLICIT OPERATOR ACTION — never runs automatically. ``None`` targets mean
    "revert every applied migration" (the pre-chain state, revisions NULL, so
    the next startup re-applies the whole chain). Mirrors
    :func:`run_database_migrations`: same per-database locking, both
    access-control modes, per-database failure isolation. Raises inside a
    database's span are reported as ``failed`` for that database; a migration
    without a ``down()`` in the span fails it up front (chains cannot skip).
    """
    if not backend_access_control_enabled():
        db_engine = get_relational_engine()
        try:
            async with _migration_lock(db_engine, _GLOBAL_MIGRATION_LOCK_KEY):
                async with db_engine.get_async_session() as session:
                    record = await session.get(
                        GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID
                    )
                if record is None:
                    return []
                graph_stored = record.global_graph_migration_revision
                vector_stored = record.global_vector_migration_revision

                graph_engine = await get_graph_engine()
                vector_engine = get_vector_engine()
                migration_context = MigrationContext(
                    graph_engine=graph_engine,
                    vector_engine=vector_engine,
                    dataset_id=None,
                )
                vector_reverted = await _revert(
                    migration_context, VECTOR_MIGRATIONS, vector_stored, vector_target_revision
                )
                graph_reverted = await _revert(
                    migration_context, GRAPH_MIGRATIONS, graph_stored, graph_target_revision
                )

                async with db_engine.get_async_session() as session:
                    record = await session.get(
                        GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID
                    )
                    if record is not None:
                        if graph_reverted:
                            record.global_graph_migration_revision = graph_target_revision
                        if vector_reverted:
                            record.global_vector_migration_revision = vector_target_revision
                        await session.commit()
        except Exception:
            logger.exception("Database downgrade failed for the global databases.")
            return [{"database": "global", "result": "failed"}]
        return [
            {
                "database": "global",
                "graph_migrations_reverted": graph_reverted,
                "vector_migrations_reverted": vector_reverted,
            }
        ]

    rows = await get_dataset_databases()
    db_engine = get_relational_engine()
    summaries: list[dict] = []

    for row in rows:
        try:
            summary = await _downgrade_dataset(
                db_engine, row, graph_target_revision, vector_target_revision
            )
        except Exception:
            logger.exception(
                "Database downgrade failed for dataset '%s'; continuing with remaining datasets.",
                row.dataset_id,
            )
            summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue
        if summary is None:
            continue
        summaries.append(summary)
        if summary["graph_migrations_reverted"] or summary["vector_migrations_reverted"]:
            logger.info(
                "Downgraded dataset '%s': graph=%s vector=%s.",
                row.dataset_id,
                summary["graph_migrations_reverted"],
                summary["vector_migrations_reverted"],
            )

    return summaries
