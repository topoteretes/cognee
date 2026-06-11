"""Startup runner that applies pending data migrations to every database.

- Access control ON  -> one ``dataset_database`` row per dataset carries that
  database pair's revision; each row's databases are resolved through the
  per-dataset context and migrated independently.
- Access control OFF -> a single global graph/vector pair backs every dataset,
  so the revision lives in the standalone single-row ``global_database_version``
  table (one database, one row — per-dataset tracking is meaningless here).

ONE revision chain covers everything (see ``registry.py``): migrations are
cross-store transformations, so there is one stored revision per database
pair, walked forward to head (plus, as an audit value, the current Cognee
version). The stamp is advanced after EVERY applied/reverted step, so a crash
or failure mid-chain leaves the bookkeeping pointing at exactly the last
consistent state.

Concurrency: the migrate-then-stamp sequence runs under a cross-process mutex.
On Postgres this is a session-scoped advisory lock held on a dedicated
connection with NO open transaction, so a long migration neither blocks row
access nor trips idle-in-transaction timeouts; the stored revision is re-read
after acquiring the lock, so the loser of a startup race sees the winner's
stamp and skips. On SQLite there is NO cross-process lock — running multiple
workers against one SQLite metadata store during a migration window is not
supported (the post-lock re-read narrows the race but cannot close it).
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from cognee.infrastructure.databases.exceptions import EntityNotFoundError

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
    head_revision,
    migrations_to_downgrade,
    pending_migrations,
)
from cognee.modules.migrations.models import GLOBAL_DATABASE_VERSION_ROW_ID, GlobalDatabaseVersion
from cognee.modules.migrations.registry import MIGRATIONS

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
    context: MigrationContext,
    stored_revision: Optional[str],
    target_revision: str = "head",
    stamp=None,
) -> list[str]:
    """Run every migration pending up to ``target_revision``, in order,
    stamping after each step via ``stamp(new_revision)`` when provided."""
    pending = pending_migrations(MIGRATIONS, stored_revision, target_revision)
    applied = []
    for migration in pending:
        logger.info(
            "Applying migration '%s' (cognee %s).", migration.slug, migration.cognee_version
        )
        await migration.up(context)
        if stamp is not None:
            await stamp(migration.revision)
        applied.append(migration.slug)
    return applied


def _downgrade_span(
    stored_revision: Optional[str], target_revision: Optional[str]
) -> list[Migration]:
    """Validate and return the migrations to revert (may be []).

    Raises (unknown stored revision, irreversible span, target ahead of
    stored) BEFORE anything executes.
    """
    return migrations_to_downgrade(MIGRATIONS, stored_revision, target_revision)


async def _revert_span(context: MigrationContext, span: list[Migration], stamp) -> list[str]:
    """Run the down() of every migration in the (pre-validated) span, newest
    first, STAMPING AFTER EACH STEP via ``stamp(new_revision)``.

    Per-step stamping means a crash or failure mid-span leaves the stored
    revision pointing at exactly the last consistent state — never at a
    revision whose data has already been reverted.
    """
    reverted = []
    for migration in span:
        logger.info("Reverting migration '%s'.", migration.slug)
        await migration.down(context)
        await stamp(migration.down_revision)
        reverted.append(migration.slug)
    return reverted


async def _stamp_dataset(db_engine, dataset_id: UUID, revision: Optional[str]) -> None:
    """Write the revision on a dataset_database row (short transaction)."""
    async with db_engine.get_async_session() as session:
        record = await session.get(DatasetDatabase, dataset_id)
        if record is not None:
            record.migration_revision = revision
            await session.commit()


async def _stamp_global(db_engine, revision: Optional[str]) -> None:
    """Write the revision on the global_database_version row."""
    async with db_engine.get_async_session() as session:
        record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        if record is not None:
            record.global_migration_revision = revision
            await session.commit()


async def _record_deployment_version(current_version: str) -> GlobalDatabaseVersion:
    """Upsert the single ``global_database_version`` row's ``cognee_version``.

    Runs on every startup in BOTH access-control modes, so this row is the one
    place to read which Cognee release last ran. The global revision column is
    left untouched here (NULL on creation): in per-dataset mode it stays NULL
    forever, and in global mode NULL means "run every migration" — correct for
    an upgrade, and a free no-op chain run followed by a head stamp for a
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


async def _run_global_migrations(current_version: str, target: str = "head") -> list[dict]:
    """Migrate the single global graph/vector pair (access control disabled).

    Same lock + re-check + migrate + per-step stamp sequence as the
    per-dataset path, against the ``global_database_version`` row. The
    migration context carries ``dataset_id=None``, so ledger updates apply
    unscoped — correct here, since the one global graph backs every dataset's
    ledger rows.
    """
    row = await _record_deployment_version(current_version)
    if not pending_migrations(MIGRATIONS, row.global_migration_revision, target):
        return [{"database": "global", "migrations_applied": []}]

    db_engine = get_relational_engine()
    try:
        async with _migration_lock(db_engine, _GLOBAL_MIGRATION_LOCK_KEY):
            # Re-read under the lock: a concurrent worker may have finished.
            async with db_engine.get_async_session() as session:
                record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is None or not pending_migrations(
                MIGRATIONS, record.global_migration_revision, target
            ):
                return []
            stored = record.global_migration_revision

            async def stamp(revision):
                await _stamp_global(db_engine, revision)

            # No context override: without access control, get_graph_engine /
            # get_vector_engine resolve the global databases directly.
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=None,
            )
            applied = await _apply(migration_context, stored, target, stamp)
    except Exception:
        logger.exception("Database migrations failed for the global databases.")
        return [{"database": "global", "result": "failed"}]

    if applied:
        logger.info("Migrated global databases: %s.", applied)
    return [{"database": "global", "migrations_applied": applied}]


async def _migrate_dataset(db_engine, row, current_version: str, target: str) -> Optional[dict]:
    """Run pending migrations for one dataset's database pair, under its lock.

    Relational transactions stay SHORT: one read after acquiring the lock, the
    migration itself runs with nothing relational open, a per-step stamp write
    after each applied migration. Returns a summary dict, or ``None`` when
    there was nothing to do (or the row vanished — dataset deleted
    concurrently).
    """
    async with _migration_lock(db_engine, _advisory_key(row.dataset_id)):
        # Re-read under the lock: a concurrent worker may have finished.
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                return None
            stored = record.migration_revision
        if not pending_migrations(MIGRATIONS, stored, target):
            return None

        async def stamp(revision):
            await _stamp_dataset(db_engine, row.dataset_id, revision)

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
            applied = await _apply(migration_context, stored, target, stamp)

        if applied:
            # Audit only: the release that last migrated this dataset.
            async with db_engine.get_async_session() as session:
                record = await session.get(DatasetDatabase, row.dataset_id)
                if record is not None:
                    record.cognee_version = current_version
                    await session.commit()

    return {"dataset_id": str(row.dataset_id), "migrations_applied": applied}


async def run_database_migrations(target: str = "head") -> list[dict]:
    """Apply pending data migrations to every Cognee database.

    ``target`` is ``"head"`` (default) or a slug (alembic-style partial
    upgrade up to and including it). Failures for one database are logged and
    skipped so the remaining databases are still migrated. Returns a
    per-database summary.
    """
    current_version = get_cognee_version()

    if not backend_access_control_enabled():
        return await _run_global_migrations(current_version, target)

    try:
        # Record the deployment-wide version even in per-dataset mode (the
        # global revision column stays NULL — per-dataset revisions live on
        # each row).
        await _record_deployment_version(current_version)
        rows = await get_dataset_databases()
    except (OperationalError, ProgrammingError, EntityNotFoundError) as error:
        # The bookkeeping tables may not exist yet on a fresh database. A
        # missing table surfaces as OperationalError on SQLite ("no such
        # table") and as ProgrammingError on PostgreSQL/asyncpg
        # (UndefinedTableError); skip the migrations in both cases (the
        # startup path runs Alembic first, so this only affects direct calls).
        logger.warning(
            "Skipping graph/vector migrations. Could not access migration bookkeeping tables: %s",
            error,
        )
        return []

    db_engine = get_relational_engine()
    summaries: list[dict] = []

    for row in rows:
        # Fast path: nothing pending per this row's snapshot — skip without
        # locking, opening the dataset's databases, or writing anything.
        if not pending_migrations(MIGRATIONS, row.migration_revision, target):
            summaries.append({"dataset_id": str(row.dataset_id), "migrations_applied": []})
            continue

        try:
            summary = await _migrate_dataset(db_engine, row, current_version, target)
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
        if summary["migrations_applied"]:
            logger.info("Migrated dataset '%s': %s.", row.dataset_id, summary["migrations_applied"])

    return summaries


async def _downgrade_dataset(db_engine, row, target: Optional[str]) -> Optional[dict]:
    """Revert migrations for one dataset's database pair, under its lock.

    Fast path first: the span is computed from the row snapshot, so datasets
    with nothing to revert are skipped without locking or resolving engines.
    """
    # Fast path (validates too: a bad stored revision fails loudly here).
    if not _downgrade_span(row.migration_revision, target):
        return None

    async with _migration_lock(db_engine, _advisory_key(row.dataset_id)):
        # Re-read under the lock: a concurrent operator may have finished.
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                return None
            stored = record.migration_revision

        span = _downgrade_span(stored, target)
        if not span:
            return None

        async def stamp(revision):
            await _stamp_dataset(db_engine, row.dataset_id, revision)

        async with set_database_global_context_variables(row.dataset_id, row.owner_id):
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=row.dataset_id,
            )
            reverted = await _revert_span(migration_context, span, stamp)

    return {"dataset_id": str(row.dataset_id), "migrations_reverted": reverted}


async def downgrade_database_migrations(
    target_revision: Optional[str] = None,
    dataset_ids: Optional[list[UUID]] = None,
) -> list[dict]:
    """Revert data migrations back to ``target_revision``.

    EXPLICIT OPERATOR ACTION — never runs automatically. ``None`` target means
    "revert every applied migration" (the pre-chain state, revision NULL, so
    the next startup re-applies the whole chain). ``dataset_ids`` restricts
    the operation to specific datasets (per-dataset mode only). The span is
    validated before any down() executes, and the stored revision is stamped
    after EVERY reverted step, so a failure mid-span leaves bookkeeping
    consistent with the data.
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
                span = _downgrade_span(record.global_migration_revision, target_revision)
                if not span:
                    return [{"database": "global", "migrations_reverted": []}]

                async def stamp(revision):
                    await _stamp_global(db_engine, revision)

                graph_engine = await get_graph_engine()
                vector_engine = get_vector_engine()
                migration_context = MigrationContext(
                    graph_engine=graph_engine,
                    vector_engine=vector_engine,
                    dataset_id=None,
                )
                reverted = await _revert_span(migration_context, span, stamp)
        except Exception:
            logger.exception("Database downgrade failed for the global databases.")
            return [{"database": "global", "result": "failed"}]
        return [{"database": "global", "migrations_reverted": reverted}]

    rows = await get_dataset_databases()
    if dataset_ids is not None:
        wanted = set(dataset_ids)
        rows = [row for row in rows if row.dataset_id in wanted]
    db_engine = get_relational_engine()
    summaries: list[dict] = []

    for row in rows:
        try:
            summary = await _downgrade_dataset(db_engine, row, target_revision)
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
        if summary["migrations_reverted"]:
            logger.info(
                "Downgraded dataset '%s': %s.", row.dataset_id, summary["migrations_reverted"]
            )

    return summaries


async def stamp_revisions(
    target: str,
    dataset_ids: Optional[list[UUID]] = None,
) -> list[dict]:
    """Set the stored revision WITHOUT running any migration (alembic `stamp`).

    EXPLICIT OPERATOR ACTION for repairing bookkeeping that has drifted from
    reality — e.g. a restored graph/vector backup behind a head-stamped row
    (stamp 'base', then upgrade re-runs the idempotent chain), or data
    verified migrated by hand. ``target``: ``"head"``, ``"base"`` (-> NULL),
    or a slug. Validates slugs against the chain; never touches data.
    """
    if target == "head":
        revision = head_revision(MIGRATIONS)
    elif target == "base":
        revision = None
    else:
        revisions = [migration.revision for migration in MIGRATIONS]
        if target not in revisions:
            raise ValueError(f"Revision {target!r} is unknown to the chain; cannot stamp.")
        revision = target

    db_engine = get_relational_engine()
    summaries: list[dict] = []

    if not backend_access_control_enabled():
        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is None:
                return []
            record.global_migration_revision = revision
            await session.commit()
        return [{"database": "global", "revision": revision}]

    rows = await get_dataset_databases()
    if dataset_ids is not None:
        wanted = set(dataset_ids)
        rows = [row for row in rows if row.dataset_id in wanted]

    for row in rows:
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                continue
            record.migration_revision = revision
            await session.commit()
        summaries.append({"dataset_id": str(row.dataset_id), "revision": revision})

    return summaries
