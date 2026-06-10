"""Startup runner that applies pending graph/vector migrations to every dataset database.

Requires backend access control (one ``dataset_database`` row per dataset, which
is where revisions are tracked); when it is disabled there is no bookkeeping row,
so the runner is a no-op.

For each row the runner walks its stored revision forward to head (see
``migration.py``) and records the new head revision and the current Cognee
version. The migrate-then-stamp sequence runs under a row lock
(``SELECT ... FOR UPDATE``) so concurrent workers starting at once serialize per
database instead of running the same data-rewriting migration twice; SQLite
ignores ``FOR UPDATE`` but serializes writers at the database level, and the
re-check of the stored revision under the lock closes the remaining window.
"""

import logging

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


def _nothing_pending(row: DatasetDatabase) -> bool:
    """True when both revision chains are already satisfied by the row's snapshot."""
    return not pending_migrations(
        GRAPH_MIGRATIONS, row.graph_migration_revision
    ) and not pending_migrations(VECTOR_MIGRATIONS, row.vector_migration_revision)


async def run_database_migrations() -> list[dict]:
    """Apply pending graph and vector migrations to every Cognee dataset database.

    Failures for one database are logged and skipped so the remaining databases
    are still migrated. Returns a per-database summary.
    """
    if not backend_access_control_enabled():
        logger.info(
            "Backend access control is disabled; no per-dataset database rows exist, "
            "skipping graph/vector migrations."
        )
        return []

    current_version = get_cognee_version()
    rows = await get_dataset_databases()
    db_engine = get_relational_engine()

    summaries: list[dict] = []

    for row in rows:
        # Fast path: nothing pending per this row's snapshot — skip without
        # opening the dataset's databases or writing anything.
        if _nothing_pending(row):
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
                if record is None or _nothing_pending(record):
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
