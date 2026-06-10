"""Startup runner that applies pending graph/vector migrations to every database.

- Access control ON  -> one ``dataset_database`` row per dataset carries that
  database pair's revisions; each row's databases are resolved through the
  per-dataset context and migrated independently.
- Access control OFF -> a single global graph/vector pair backs every dataset,
  so revisions live in the standalone single-row ``global_database_version``
  table (one database, one row — per-dataset tracking is meaningless here).

In both modes the runner walks the stored revision forward to head (see
``migration.py``) and records the new head revision and the current Cognee
version. The migrate-then-stamp sequence runs under a row lock
(``SELECT ... FOR UPDATE``) so concurrent workers starting at once serialize per
database instead of running the same data-rewriting migration twice; SQLite
ignores ``FOR UPDATE`` but serializes writers at the database level, and the
re-check of the stored revision under the lock closes the remaining window.
"""

import logging

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
    pending_migrations,
)
from cognee.modules.migrations.models import GLOBAL_DATABASE_VERSION_ROW_ID, GlobalDatabaseVersion
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

logger = logging.getLogger(__name__)


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
        async with db_engine.get_async_session() as session:
            record = await session.get(
                GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID, with_for_update=True
            )
            if record is None or _nothing_pending(
                record.global_graph_migration_revision, record.global_vector_migration_revision
            ):
                return []

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
                migration_context, GRAPH_MIGRATIONS, record.global_graph_migration_revision
            )
            vector_applied, vector_revision = await _apply(
                migration_context, VECTOR_MIGRATIONS, record.global_vector_migration_revision
            )

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
        # opening the dataset's databases or writing anything.
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
            async with db_engine.get_async_session() as session:
                # Lock the row and re-read its revisions: a concurrent worker
                # may have migrated this database while we waited for the lock.
                record = await session.get(DatasetDatabase, row.dataset_id, with_for_update=True)
                if record is None or _nothing_pending(
                    record.graph_migration_revision, record.vector_migration_revision
                ):
                    continue

                # Resolve this dataset's graph/vector databases through the
                # per-dataset context.
                async with set_database_global_context_variables(
                    record.dataset_id, record.owner_id
                ):
                    graph_engine = await get_graph_engine()
                    vector_engine = get_vector_engine()
                    # One context carries every store a migration may need to
                    # touch (graph + vector + relational ledger).
                    migration_context = MigrationContext(
                        graph_engine=graph_engine,
                        vector_engine=vector_engine,
                        dataset_id=record.dataset_id,
                    )
                    graph_applied, graph_revision = await _apply(
                        migration_context, GRAPH_MIGRATIONS, record.graph_migration_revision
                    )
                    vector_applied, vector_revision = await _apply(
                        migration_context, VECTOR_MIGRATIONS, record.vector_migration_revision
                    )

                record.graph_migration_revision = graph_revision
                record.vector_migration_revision = vector_revision
                record.cognee_version = current_version
                await session.commit()
        except Exception:
            logger.exception(
                "Database migrations failed for dataset '%s'; continuing with remaining datasets.",
                row.dataset_id,
            )
            summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue

        summaries.append(
            {
                "dataset_id": str(row.dataset_id),
                "graph_migrations_applied": graph_applied,
                "vector_migrations_applied": vector_applied,
            }
        )
        if graph_applied or vector_applied:
            logger.info(
                "Migrated dataset '%s': graph=%s vector=%s.",
                row.dataset_id,
                graph_applied,
                vector_applied,
            )

    return summaries
