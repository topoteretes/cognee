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

Concurrency: the migrate-then-stamp sequence runs under a cross-process mutex
(see ``_migration_lock``), and the stored revision is re-read after acquiring
it, so the loser of a startup race sees the winner's stamp and skips. On
Postgres the mutex is a session-scoped advisory lock on a dedicated connection
with NO open transaction (also serializes across hosts). On SQLite it is an OS
advisory file lock (``filelock``) next to the database — serializing multiple
processes on ONE host (multi-worker servers, parallel SDK runs), but not across
hosts / NFS, for which Postgres metadata is required.
"""

import asyncio
import concurrent.futures
import logging
import os
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
from cognee.infrastructure.databases.vector import get_vector_engine_async
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
from cognee.modules.migrations.versions.adapter_storage_migration import (
    migrate as _run_adapter_storage_migration,
)

logger = logging.getLogger(__name__)

# The ONE migration mutex key. Every migration critical section — the relational
# schema bootstrap (alembic upgrade / create+stamp), the global database-pair
# migration, and each per-dataset migration — takes this same key, so a host runs
# at most one migration step at a time across all processes. Each is a separate,
# sequential acquisition; they never nest, so the shared key cannot self-deadlock.
_GLOBAL_MIGRATION_LOCK_KEY = 0x636F676E6565_01  # "cognee" + 01


def _file_lock_path(db_engine, key: int) -> Optional[str]:
    """Lock-file path next to the SQLite database, one per ``key``.

    All processes on the same store resolve the same file for a given ``key``
    (always the single global migration key here). ``None`` for an in-memory DB —
    nothing to coordinate.
    """
    db_path = db_engine.engine.url.database
    if not db_path or db_path == ":memory:":
        return None
    directory = os.path.dirname(os.path.abspath(db_path))
    return os.path.join(directory, f".cognee-migration-{key}.lock")


@asynccontextmanager
async def _migration_lock(db_engine, key: int):
    """Cross-process mutex around the migrate-then-stamp sequence.

    Postgres: a session-scoped ``pg_advisory_lock`` (committed right away, since
    it survives commit) — also serializes across hosts. SQLite: an OS advisory
    file lock (``filelock``, portable across Linux/macOS/Windows), auto-released
    if the holder crashes. The file lock serializes processes on one host
    (multi-worker servers, parallel SDK runs) but not across hosts/NFS — use
    Postgres metadata for multi-host.

    Not re-entrant: a caller must never acquire the same key twice (re-locking it
    on a second connection self-deadlocks). The single global key is taken at one
    level only — see run_migrations / run_database_migrations.
    """
    engine = db_engine.engine
    if engine.dialect.name == "postgresql":
        async with engine.connect() as connection:
            await connection.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
            await connection.commit()
            try:
                yield
            finally:
                await connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                await connection.commit()
        return

    lock_path = _file_lock_path(db_engine, key)
    if lock_path is None:
        # In-memory / pathless DB: not shareable across processes anyway.
        yield
        return

    from filelock import FileLock

    # Acquire and release MUST run on the same OS thread, off the event loop.
    # filelock keeps two pieces of per-thread state that break if they don't:
    # its deadlock-detection registry (a threading.local mapping lock path ->
    # holder instance) is written by acquire and popped by release each on the
    # thread they run on — a release on another thread leaves a ghost entry on
    # the acquiring thread, and the NEXT acquisition landing there fails with a
    # false "Deadlock: lock is already held by a different FileLock instance".
    # asyncio.to_thread assigns pool threads arbitrarily, so instead run both
    # calls on one dedicated short-lived thread. Each acquisition gets its own
    # executor, so concurrent in-process acquisitions still block one another
    # via the OS lock (each waiting on its own thread) — the same mutual
    # exclusion the Postgres advisory-lock branch provides.
    # thread_local=False keeps the instance's re-entrancy counter shared across
    # threads so a release still works even if thread affinity ever regresses.
    lock = FileLock(lock_path, thread_local=False)
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="cognee-migration-lock"
    )
    try:
        await loop.run_in_executor(executor, lock.acquire)
        try:
            yield
        finally:
            await loop.run_in_executor(executor, lock.release)
    finally:
        executor.shutdown(wait=False)


@asynccontextmanager
async def migration_lock():
    """THE one global migration lock — acquire around any migration flow.

    Every entry point that migrates (startup bootstrap, the CLI ``migrate`` /
    ``downgrade`` commands) wraps its whole sequence — relational schema bootstrap
    AND the graph/vector data migrations — in this single mutex, keyed on
    ``_GLOBAL_MIGRATION_LOCK_KEY`` against the relational engine. So a host runs at
    most ONE migration of any kind at a time across all processes. The runner
    functions (run_database_migrations / downgrade_database_migrations and their
    per-dataset/global helpers) do NOT lock themselves — they assume this is held —
    so there is exactly one acquisition per migration run, never one per dataset
    and never nested.
    """
    async with _migration_lock(get_relational_engine(), _GLOBAL_MIGRATION_LOCK_KEY):
        yield


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


def _error_text(error: Exception) -> str:
    """Compact, persistable description of a migration failure."""
    return f"{type(error).__name__}: {error}"[:500]


async def _record_dataset_failure(db_engine, dataset_id: UUID, error: Exception) -> None:
    """Persist why this dataset's migration failed (cleared on next success)."""
    try:
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, dataset_id)
            if record is not None:
                record.migration_last_error = _error_text(error)
                await session.commit()
    except Exception:  # noqa: BLE001 - never let bookkeeping mask the real failure
        logger.exception("Could not persist migration failure for dataset '%s'.", dataset_id)


async def _record_global_failure(db_engine, error: Exception) -> None:
    """Persist why the global migration failed (cleared on next success)."""
    try:
        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is not None:
                record.global_migration_last_error = _error_text(error)
                await session.commit()
    except Exception:  # noqa: BLE001 - never let bookkeeping mask the real failure
        logger.exception("Could not persist global migration failure.")


async def _read_deployment_version() -> Optional[str]:
    """The deployment's recorded Cognee version, or ``None`` if never recorded.

    Read BEFORE ``_record_deployment_version`` overwrites it, so a version change
    is detectable — the gate for the vector adapter storage sync (which, unlike
    the once-per-database chain, must run on every version change).
    """
    db_engine = get_relational_engine()
    try:
        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            return record.cognee_version if record is not None else None
    except (OperationalError, ProgrammingError, EntityNotFoundError):
        return None


async def _sync_vector_adapter_storage(migration_context: MigrationContext) -> None:
    """Sync the vector adapter's stored schema (e.g. LanceDB columns) to current.

    Idempotent; run only on a version change, after the chain. A failure
    propagates like any migration failure for this database (the caller records
    it and blocks writes).
    """
    await _run_adapter_storage_migration(migration_context)


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


async def _run_global_migrations(
    current_version: str, version_changed: bool, target: str = "head"
) -> list[dict]:
    """Migrate the single global graph/vector pair (access control disabled).

    The caller holds the single migration lock. Re-check + migrate + per-step
    stamp against the ``global_database_version`` row, with ``dataset_id=None`` so
    ledger updates apply unscoped (one global graph backs every dataset).
    ``version_changed`` also runs the adapter storage sync after the chain, even
    with no chain migration pending.
    """
    try:
        row = await _record_deployment_version(current_version)
    except (OperationalError, ProgrammingError, EntityNotFoundError) as error:
        # Same missing-bookkeeping-table tolerance as the per-dataset branch.
        logger.warning(
            "Skipping graph/vector migrations. Could not access migration bookkeeping tables: %s",
            error,
        )
        return []
    if (
        not pending_migrations(MIGRATIONS, row.global_migration_revision, target)
        and not version_changed
    ):
        return [{"database": "global", "migrations_applied": []}]

    db_engine = get_relational_engine()
    try:
        # Re-read under the held lock: a concurrent worker may have finished.
        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        if record is None or (
            not pending_migrations(MIGRATIONS, record.global_migration_revision, target)
            and not version_changed
        ):
            return [{"database": "global", "migrations_applied": []}]
        stored = record.global_migration_revision

        async def stamp(revision):
            await _stamp_global(db_engine, revision)

        # No context override: without access control, get_graph_engine /
        # get_vector_engine_async resolve the global databases directly.
        graph_engine = await get_graph_engine()
        vector_engine = await get_vector_engine_async()
        migration_context = MigrationContext(
            graph_engine=graph_engine,
            vector_engine=vector_engine,
            dataset_id=None,
        )
        applied = await _apply(migration_context, stored, target, stamp)
        # Version bumped -> sync the vector adapter's stored schema, last.
        if version_changed:
            await _sync_vector_adapter_storage(migration_context)
        if applied or version_changed:
            async with db_engine.get_async_session() as session:
                record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
                if record is not None and record.global_migration_last_error is not None:
                    record.global_migration_last_error = None
                    await session.commit()
    except Exception as error:
        logger.exception("Database migrations failed for the global databases.")
        await _record_global_failure(db_engine, error)
        return [{"database": "global", "result": "failed"}]

    if applied:
        logger.info("Migrated global databases: %s.", applied)
    return [{"database": "global", "migrations_applied": applied}]


async def _migrate_dataset(
    db_engine, row, current_version: str, version_changed: bool, target: str
) -> Optional[dict]:
    """Run pending migrations for one dataset's database pair.

    The caller (run_database_migrations) holds the single migration lock for the
    whole dataset loop, so this does not lock itself. Relational transactions stay
    short: read, migrate with nothing relational open, stamp after each step.
    Returns a summary, or ``None`` when there was nothing to do (or the dataset was
    deleted concurrently). ``version_changed`` also runs the adapter storage sync
    for this dataset's vector DB after the chain, even with no chain migration
    pending.
    """
    # Re-read under the held lock: a concurrent worker may have finished.
    async with db_engine.get_async_session() as session:
        record = await session.get(DatasetDatabase, row.dataset_id)
        if record is None:
            return None
        stored = record.migration_revision
    if not pending_migrations(MIGRATIONS, stored, target) and not version_changed:
        return None

    async def stamp(revision):
        await _stamp_dataset(db_engine, row.dataset_id, revision)

    # Resolve this dataset's graph/vector databases through the
    # per-dataset context — the same way every other operation does.
    async with set_database_global_context_variables(row.dataset_id, row.owner_id):
        graph_engine = await get_graph_engine()
        vector_engine = await get_vector_engine_async()
        migration_context = MigrationContext(
            graph_engine=graph_engine,
            vector_engine=vector_engine,
            dataset_id=row.dataset_id,
        )
        applied = await _apply(migration_context, stored, target, stamp)
        # Version bumped -> sync this dataset's vector adapter schema, last.
        if version_changed:
            await _sync_vector_adapter_storage(migration_context)

    if applied or version_changed:
        try:
            # Audit/health bookkeeping: the release that last migrated this
            # dataset, and clearing any recorded failure. Best-effort — the
            # chain is already applied and stamped; a hiccup here must not
            # report the migration itself as failed.
            async with db_engine.get_async_session() as session:
                record = await session.get(DatasetDatabase, row.dataset_id)
                if record is not None:
                    record.cognee_version = current_version
                    record.migration_last_error = None
                    await session.commit()
        except Exception:  # noqa: BLE001 - audit only
            logger.exception(
                "Could not record audit fields for dataset '%s' (migrations applied fine).",
                row.dataset_id,
            )

    return {"dataset_id": str(row.dataset_id), "migrations_applied": applied}


async def run_database_migrations(target: str = "head") -> list[dict]:
    """Apply pending data migrations to every Cognee database.

    ``target`` is ``"head"`` (default) or a slug (alembic-style partial
    upgrade up to and including it). Failures for one database are logged and
    skipped so the remaining databases are still migrated. Returns a
    per-database summary.

    The CALLER must hold the single migration lock (see ``_migration_lock``):
    ``run_migrations`` wraps the relational bootstrap + this call in it,
    and the CLI ``migrate`` command acquires it around this call. This function
    does not lock, so a host runs at most one migration — relational, global, and
    every dataset — under one mutex, never one lock per dataset.
    """
    current_version = get_cognee_version()

    # Validate the target up front: an unknown slug is an operator error and
    # must fail the whole call fast, never as N per-dataset failures.
    pending_migrations(MIGRATIONS, None, target)

    # Read the deployment's recorded version BEFORE it is overwritten below, so
    # a Cognee version change is detectable. It gates the vector adapter storage
    # sync, which (unlike the once-per-database revision chain) must run on every
    # version change — even a release that ships no data migration.
    version_changed = await _read_deployment_version() != current_version

    if not backend_access_control_enabled():
        return await _run_global_migrations(current_version, version_changed, target)

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
        # Fast path: nothing pending AND no version change for this row — skip
        # without opening the dataset's databases or writing anything.
        if (
            not pending_migrations(MIGRATIONS, row.migration_revision, target)
            and not version_changed
        ):
            summaries.append({"dataset_id": str(row.dataset_id), "migrations_applied": []})
            continue

        try:
            summary = await _migrate_dataset(
                db_engine, row, current_version, version_changed, target
            )
        except Exception as error:
            logger.exception(
                "Database migrations failed for dataset '%s'; continuing with the rest.",
                row.dataset_id,
            )
            await _record_dataset_failure(db_engine, row.dataset_id, error)
            summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue

        if summary is None:
            continue
        summaries.append(summary)
        if summary["migrations_applied"]:
            logger.info("Migrated dataset '%s': %s.", row.dataset_id, summary["migrations_applied"])

    return summaries


async def _downgrade_dataset(db_engine, row, target: Optional[str]) -> Optional[dict]:
    """Revert migrations for one dataset's database pair.

    The caller holds the single migration lock for the whole dataset loop. Fast
    path first: the span is computed from the row snapshot, so datasets with
    nothing to revert are skipped without resolving engines.
    """
    # Fast path (validates too: a bad stored revision fails loudly here).
    if not _downgrade_span(row.migration_revision, target):
        return None

    # Re-read under the held lock: a concurrent operator may have finished.
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
        vector_engine = await get_vector_engine_async()
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
    the operation to specific datasets (per-dataset mode only; raises in
    global mode rather than silently rewriting the shared pair). The span is
    validated before any down() executes, and the stored revision is stamped
    after EVERY reverted step, so a failure mid-span leaves bookkeeping
    consistent with the data.
    """
    # Validate the target up front (operator error -> fail the whole call).
    if target_revision is not None and target_revision not in (
        migration.revision for migration in MIGRATIONS
    ):
        raise ValueError(
            f"Target revision {target_revision!r} is unknown to the chain; cannot downgrade."
        )

    if not backend_access_control_enabled():
        if dataset_ids is not None:
            raise ValueError(
                "dataset_ids targeting requires backend access control; with it "
                "disabled there is one GLOBAL database pair shared by every dataset."
            )
        db_engine = get_relational_engine()
        try:
            async with db_engine.get_async_session() as session:
                record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is None:
                return []
            span = _downgrade_span(record.global_migration_revision, target_revision)
            if not span:
                return [{"database": "global", "migrations_reverted": []}]

            async def stamp(revision):
                await _stamp_global(db_engine, revision)

            graph_engine = await get_graph_engine()
            vector_engine = await get_vector_engine_async()
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
        missing = wanted - {row.dataset_id for row in rows}
        if missing:
            raise ValueError(
                "No dataset_database row found for dataset id(s): "
                + ", ".join(str(dataset_id) for dataset_id in sorted(missing))
            )
        rows = [row for row in rows if row.dataset_id in wanted]
    db_engine = get_relational_engine()
    summaries: list[dict] = []

    for row in rows:
        try:
            summary = await _downgrade_dataset(db_engine, row, target_revision)
        except Exception:
            logger.exception(
                "Database downgrade failed for dataset '%s'; continuing with the rest.",
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
        if dataset_ids is not None:
            raise ValueError(
                "dataset_ids targeting requires backend access control; with it "
                "disabled there is one GLOBAL database pair shared by every dataset."
            )
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
        missing = wanted - {row.dataset_id for row in rows}
        if missing:
            raise ValueError(
                "No dataset_database row found for dataset id(s): "
                + ", ".join(str(dataset_id) for dataset_id in sorted(missing))
            )
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
