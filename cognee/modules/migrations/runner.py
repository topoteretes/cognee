"""Startup runner that applies pending graph/vector migrations to every database.

- Access control ON  -> one ``dataset_database`` row per dataset; each row's
  databases are resolved through the per-dataset context and migrated.
- Access control OFF -> a single reserved global ``dataset_database`` row anchors
  the shared global graph/vector databases.

For each database the runner walks its stored revision forward to head (see
``migration.py``), then records the new head revision and the current Cognee
version.
"""

import logging
from contextlib import nullcontext
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError

from cognee.version import get_cognee_version
from cognee.context_global_variables import (
    backend_access_control_enabled,
    set_database_global_context_variables,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.vector import get_vector_engine, get_vectordb_config
from cognee.modules.data.models import Dataset
from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import DatasetDatabase

from cognee.modules.migrations.constants import GLOBAL_DATASET_ID, GLOBAL_DATASET_NAME
from cognee.modules.migrations.migration import Migration, head_revision, pending_migrations
from cognee.modules.migrations.graph_migrations import GRAPH_MIGRATIONS
from cognee.modules.migrations.vector_migrations import VECTOR_MIGRATIONS

logger = logging.getLogger(__name__)


async def _apply(
    engine: Any, migrations: list[Migration], stored_revision: Optional[str]
) -> tuple[list[str], Optional[str]]:
    """Run every pending migration in order.

    Returns the applied migration slugs and the resulting revision (the chain
    head if anything ran, otherwise the unchanged stored revision).
    """
    pending = pending_migrations(migrations, stored_revision)
    for migration in pending:
        logger.info(
            "Applying migration '%s' (cognee %s).", migration.slug, migration.cognee_version
        )
        await migration.up(engine)

    new_revision = pending[-1].revision if pending else stored_revision
    return [migration.slug for migration in pending], new_revision


async def get_or_create_global_dataset_database() -> DatasetDatabase:
    """Return the reserved global ``dataset_database`` row, creating it if absent.

    Used when backend access control is disabled. The row is anchored to a
    reserved dataset (to satisfy the ``dataset_id`` foreign key) and populated
    from the global graph/vector config.

    On first creation the graph database is probed with ``is_empty()``: a fresh
    (empty) database is stamped at head so no migrations run, while a populated
    pre-existing database is left with NULL revisions so every migration runs.
    """
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        existing = await session.get(DatasetDatabase, GLOBAL_DATASET_ID)
        if existing:
            return existing

    default_user = await get_default_user()

    # Fresh (empty) graph DB -> stamp head (skip migrations). Pre-existing DB ->
    # leave revisions NULL so all migrations run.
    graph_engine = await get_graph_engine()
    is_fresh = await graph_engine.is_empty()

    graph_config = get_graph_config()
    vector_config = get_vectordb_config()

    async with db_engine.get_async_session() as session:
        # Ensure the reserved parent dataset exists to satisfy the FK on dataset_id.
        reserved_dataset = await session.get(Dataset, GLOBAL_DATASET_ID)
        if reserved_dataset is None:
            session.add(
                Dataset(id=GLOBAL_DATASET_ID, name=GLOBAL_DATASET_NAME, owner_id=default_user.id)
            )
            await session.flush()

        record = DatasetDatabase(
            owner_id=default_user.id,
            dataset_id=GLOBAL_DATASET_ID,
            graph_database_provider=graph_config.graph_database_provider,
            graph_database_name=graph_config.graph_database_name or GLOBAL_DATASET_NAME,
            graph_dataset_database_handler=graph_config.graph_dataset_database_handler,
            vector_database_provider=vector_config.vector_db_provider,
            vector_database_name=vector_config.vector_db_name or GLOBAL_DATASET_NAME,
            vector_dataset_database_handler=vector_config.vector_dataset_database_handler,
            graph_migration_revision=head_revision(GRAPH_MIGRATIONS) if is_fresh else None,
            vector_migration_revision=head_revision(VECTOR_MIGRATIONS) if is_fresh else None,
            cognee_version=get_cognee_version(),
        )
        try:
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record
        except IntegrityError:
            # Concurrent startup (e.g. multiple workers) already created it.
            await session.rollback()
            existing = await session.get(DatasetDatabase, GLOBAL_DATASET_ID)
            if existing is None:
                raise
            return existing


async def _store_revisions(
    dataset_id: UUID,
    graph_revision: Optional[str],
    vector_revision: Optional[str],
    cognee_version: str,
) -> None:
    """Persist the post-migration revisions and current Cognee version."""
    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        record = await session.get(DatasetDatabase, dataset_id)
        if record is None:
            return
        record.graph_migration_revision = graph_revision
        record.vector_migration_revision = vector_revision
        record.cognee_version = cognee_version
        await session.commit()


async def run_database_migrations() -> list[dict]:
    """Apply pending graph and vector migrations to every Cognee database.

    Failures for one database are logged and skipped so the remaining databases
    are still migrated. Returns a per-database summary.
    """
    access_control = backend_access_control_enabled()
    current_version = get_cognee_version()

    if access_control:
        rows = await get_dataset_databases()
    else:
        rows = [await get_or_create_global_dataset_database()]

    summaries: list[dict] = []

    for row in rows:
        # Resolve per-dataset databases through the context manager when access
        # control is on; the global databases need no context override.
        context = (
            set_database_global_context_variables(row.dataset_id, row.owner_id)
            if access_control
            else nullcontext()
        )
        try:
            async with context:
                graph_engine = await get_graph_engine()
                vector_engine = get_vector_engine()
                graph_applied, graph_revision = await _apply(
                    graph_engine, GRAPH_MIGRATIONS, row.graph_migration_revision
                )
                vector_applied, vector_revision = await _apply(
                    vector_engine, VECTOR_MIGRATIONS, row.vector_migration_revision
                )
        except Exception:
            logger.exception(
                "Database migrations failed for dataset '%s'; continuing with remaining datasets.",
                row.dataset_id,
            )
            summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue

        await _store_revisions(row.dataset_id, graph_revision, vector_revision, current_version)
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
