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
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

logger = logging.getLogger(__name__)

# Target sentinel: leave a chain completely untouched (used when an explicit
# revision was given for the OTHER chain, alembic-style partial operations).
KEEP = "keep"

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
    migrations: list[Migration],
    stored_revision: str | None,
    target_revision: str = "head",
) -> tuple[list[str], str | None]:
    """Run every migration pending up to ``target_revision`` in order.

    Returns the applied migration slugs and the resulting revision (the last
    applied if anything ran, otherwise the unchanged stored revision).
    """
    if target_revision == KEEP:
        return [], stored_revision
    pending = pending_migrations(migrations, stored_revision, target_revision)
    for migration in pending:
        logger.info(
            "Applying migration '%s' (cognee %s).", migration.slug, migration.cognee_version
        )
        await migration.up(context)

    new_revision = pending[-1].revision if pending else stored_revision
    return [migration.slug for migration in pending], new_revision


def _nothing_pending(
    graph_revision: str | None,
    vector_revision: str | None,
    graph_target: str = "head",
    vector_target: str = "head",
) -> bool:
    """True when both revision chains are already satisfied by the given snapshot."""
    graph_pending = (
        []
        if graph_target == KEEP
        else pending_migrations(GRAPH_MIGRATIONS, graph_revision, graph_target)
    )
    vector_pending = (
        []
        if vector_target == KEEP
        else pending_migrations(VECTOR_MIGRATIONS, vector_revision, vector_target)
    )
    return not graph_pending and not vector_pending


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


async def _run_global_migrations(
    current_version: str, graph_target: str = "head", vector_target: str = "head"
) -> list[dict]:
    """Migrate the single global graph/vector pair (access control disabled).

    Same lock + re-check + migrate + stamp sequence as the per-dataset path,
    against the ``global_database_version`` row. The migration context carries
    ``dataset_id=None``, so ledger updates apply unscoped — correct here, since
    the one global graph backs every dataset's ledger rows.
    """
    try:
        row = await _record_deployment_version(current_version)
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
    if _nothing_pending(
        row.global_graph_migration_revision,
        row.global_vector_migration_revision,
        graph_target,
        vector_target,
    ):
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
                record.global_graph_migration_revision,
                record.global_vector_migration_revision,
                graph_target,
                vector_target,
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
                migration_context, GRAPH_MIGRATIONS, graph_stored, graph_target
            )
            vector_applied, vector_revision = await _apply(
                migration_context, VECTOR_MIGRATIONS, vector_stored, vector_target
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


async def _migrate_dataset(
    db_engine, row, current_version: str, graph_target: str = "head", vector_target: str = "head"
) -> Optional[dict]:
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
        if _nothing_pending(graph_stored, vector_stored, graph_target, vector_target):
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
                migration_context, GRAPH_MIGRATIONS, graph_stored, graph_target
            )
            vector_applied, vector_revision = await _apply(
                migration_context, VECTOR_MIGRATIONS, vector_stored, vector_target
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


async def run_database_migrations(
    graph_target: str = "head", vector_target: str = "head"
) -> list[dict]:
    """Apply pending graph and vector migrations to every Cognee database.

    Targets are ``"head"`` (default), a slug (alembic-style partial upgrade up
    to and including it), or ``KEEP`` (leave that chain untouched). Failures
    for one database are logged and skipped so the remaining databases are
    still migrated. Returns a per-database summary.
    """
    current_version = get_cognee_version()

    if not backend_access_control_enabled():
        return await _run_global_migrations(current_version, graph_target, vector_target)

    try:
        # Record the deployment-wide version even in per-dataset mode (the
        # global revision columns stay NULL — per-dataset revisions live on
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
        if _nothing_pending(
            row.graph_migration_revision, row.vector_migration_revision, graph_target, vector_target
        ):
            summaries.append(
                {
                    "dataset_id": str(row.dataset_id),
                    "graph_migrations_applied": [],
                    "vector_migrations_applied": [],
                }
            )
            continue

        try:
            summary = await _migrate_dataset(
                db_engine, row, current_version, graph_target, vector_target
            )
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


def _downgrade_span(
    migrations: list[Migration],
    stored_revision: Optional[str],
    target_revision: Optional[str],
) -> list[Migration]:
    """Validate and return the migrations to revert for one chain (may be []).

    A ``KEEP`` target means the chain is untouched. Raises (unknown stored
    revision, irreversible span, target ahead of stored) BEFORE anything
    executes — both chains are validated up front so a bad graph span can
    never be discovered after vector down()s already ran.
    """
    if target_revision == KEEP:
        return []
    return migrations_to_downgrade(migrations, stored_revision, target_revision)


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


async def _downgrade_dataset(
    db_engine, row, graph_target: Optional[str], vector_target: Optional[str]
) -> Optional[dict]:
    """Revert migrations for one dataset's database pair, under its lock.

    Fast path first: spans are computed from the row snapshot, so datasets
    with nothing to revert are skipped without locking or resolving engines.
    """
    # Fast path (validates too: a bad stored revision fails loudly here).
    if not _downgrade_span(
        GRAPH_MIGRATIONS, row.graph_migration_revision, graph_target
    ) and not _downgrade_span(VECTOR_MIGRATIONS, row.vector_migration_revision, vector_target):
        return None

    async with _migration_lock(db_engine, _advisory_key(row.dataset_id)):
        # Re-read under the lock: a concurrent operator may have finished.
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                return None
            graph_stored = record.graph_migration_revision
            vector_stored = record.vector_migration_revision

        # Validate BOTH spans before executing anything from either.
        graph_span = _downgrade_span(GRAPH_MIGRATIONS, graph_stored, graph_target)
        vector_span = _downgrade_span(VECTOR_MIGRATIONS, vector_stored, vector_target)
        if not graph_span and not vector_span:
            return None

        async def stamp_graph(revision):
            await _stamp_dataset(db_engine, row.dataset_id, "graph_migration_revision", revision)

        async def stamp_vector(revision):
            await _stamp_dataset(db_engine, row.dataset_id, "vector_migration_revision", revision)

        async with set_database_global_context_variables(row.dataset_id, row.owner_id):
            graph_engine = await get_graph_engine()
            vector_engine = get_vector_engine()
            migration_context = MigrationContext(
                graph_engine=graph_engine,
                vector_engine=vector_engine,
                dataset_id=row.dataset_id,
            )
            # Reverse of the upgrade order: vector chain first, then graph.
            vector_reverted = await _revert_span(migration_context, vector_span, stamp_vector)
            graph_reverted = await _revert_span(migration_context, graph_span, stamp_graph)

    return {
        "dataset_id": str(row.dataset_id),
        "graph_migrations_reverted": graph_reverted,
        "vector_migrations_reverted": vector_reverted,
    }


async def _stamp_dataset(db_engine, dataset_id: UUID, column: str, revision: Optional[str]) -> None:
    """Write one revision column on a dataset_database row (short transaction)."""
    async with db_engine.get_async_session() as session:
        record = await session.get(DatasetDatabase, dataset_id)
        if record is not None:
            setattr(record, column, revision)
            await session.commit()


async def _stamp_global(db_engine, column: str, revision: Optional[str]) -> None:
    """Write one revision column on the global_database_version row."""
    async with db_engine.get_async_session() as session:
        record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        if record is not None:
            setattr(record, column, revision)
            await session.commit()


async def downgrade_database_migrations(
    graph_target_revision: Optional[str] = None,
    vector_target_revision: Optional[str] = None,
    dataset_ids: Optional[list[UUID]] = None,
) -> list[dict]:
    """Revert graph/vector migrations back to the given target revisions.

    EXPLICIT OPERATOR ACTION — never runs automatically. ``None`` targets mean
    "revert every applied migration" (the pre-chain state, revisions NULL, so
    the next startup re-applies the whole chain); ``KEEP`` leaves a chain
    untouched. ``dataset_ids`` restricts the operation to specific datasets
    (per-dataset mode only). Both chains' spans are validated before any
    down() executes, and the stored revision is stamped after EVERY reverted
    step, so a failure mid-span leaves bookkeeping consistent with the data.
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
                graph_span = _downgrade_span(
                    GRAPH_MIGRATIONS, record.global_graph_migration_revision, graph_target_revision
                )
                vector_span = _downgrade_span(
                    VECTOR_MIGRATIONS,
                    record.global_vector_migration_revision,
                    vector_target_revision,
                )
                if not graph_span and not vector_span:
                    return [
                        {
                            "database": "global",
                            "graph_migrations_reverted": [],
                            "vector_migrations_reverted": [],
                        }
                    ]

                async def stamp_graph(revision):
                    await _stamp_global(db_engine, "global_graph_migration_revision", revision)

                async def stamp_vector(revision):
                    await _stamp_global(db_engine, "global_vector_migration_revision", revision)

                graph_engine = await get_graph_engine()
                vector_engine = get_vector_engine()
                migration_context = MigrationContext(
                    graph_engine=graph_engine,
                    vector_engine=vector_engine,
                    dataset_id=None,
                )
                vector_reverted = await _revert_span(migration_context, vector_span, stamp_vector)
                graph_reverted = await _revert_span(migration_context, graph_span, stamp_graph)
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
    if dataset_ids is not None:
        wanted = set(dataset_ids)
        rows = [row for row in rows if row.dataset_id in wanted]
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


async def stamp_revisions(
    graph_target: str = KEEP,
    vector_target: str = KEEP,
    dataset_ids: Optional[list[UUID]] = None,
) -> list[dict]:
    """Set stored revisions WITHOUT running any migration (alembic `stamp`).

    EXPLICIT OPERATOR ACTION for repairing bookkeeping that has drifted from
    reality — e.g. a restored graph/vector backup behind a head-stamped row
    (stamp base, then upgrade re-runs the idempotent chain), or data verified
    migrated by hand. Targets: ``"head"``, ``"base"`` (-> NULL), a slug, or
    ``KEEP`` (leave that chain's stamp alone). Validates slugs against the
    chains; never touches data.
    """

    def resolve(target: str, migrations: list[Migration]) -> tuple[bool, Optional[str]]:
        if target == KEEP:
            return False, None
        if target == "head":
            return True, head_revision(migrations)
        if target == "base":
            return True, None
        revisions = [migration.revision for migration in migrations]
        if target not in revisions:
            raise ValueError(f"Revision {target!r} is unknown to this chain; cannot stamp.")
        return True, target

    set_graph, graph_revision = resolve(graph_target, GRAPH_MIGRATIONS)
    set_vector, vector_revision = resolve(vector_target, VECTOR_MIGRATIONS)
    if not set_graph and not set_vector:
        return []

    db_engine = get_relational_engine()
    summaries: list[dict] = []

    if not backend_access_control_enabled():
        async with db_engine.get_async_session() as session:
            record = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
            if record is None:
                return []
            if set_graph:
                record.global_graph_migration_revision = graph_revision
            if set_vector:
                record.global_vector_migration_revision = vector_revision
            await session.commit()
        return [
            {
                "database": "global",
                "graph_revision": graph_revision if set_graph else "(kept)",
                "vector_revision": vector_revision if set_vector else "(kept)",
            }
        ]

    rows = await get_dataset_databases()
    if dataset_ids is not None:
        wanted = set(dataset_ids)
        rows = [row for row in rows if row.dataset_id in wanted]

    for row in rows:
        async with db_engine.get_async_session() as session:
            record = await session.get(DatasetDatabase, row.dataset_id)
            if record is None:
                continue
            if set_graph:
                record.graph_migration_revision = graph_revision
            if set_vector:
                record.vector_migration_revision = vector_revision
            await session.commit()
        summaries.append(
            {
                "dataset_id": str(row.dataset_id),
                "graph_revision": graph_revision if set_graph else "(kept)",
                "vector_revision": vector_revision if set_vector else "(kept)",
            }
        )

    return summaries


def _validate_registries() -> None:
    """Cross-chain invariants, enforced at import: slugs must be unique across
    BOTH chains (a slug is a CLI revision argument and must resolve to exactly
    one chain) and must not shadow the reserved revision keywords."""
    reserved = {"head", "base", KEEP}
    graph_slugs = {migration.slug for migration in GRAPH_MIGRATIONS}
    vector_slugs = {migration.slug for migration in VECTOR_MIGRATIONS}
    overlap = graph_slugs & vector_slugs
    if overlap:
        raise ValueError(f"Migration slugs present in BOTH chains: {sorted(overlap)}")
    shadowed = (graph_slugs | vector_slugs) & reserved
    if shadowed:
        raise ValueError(f"Migration slugs shadow reserved revision keywords: {sorted(shadowed)}")


_validate_registries()
