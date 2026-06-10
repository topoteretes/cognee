import os
import sys
import logging
import subprocess
from pathlib import Path
import importlib.resources as pkg_resources

from cognee.context_global_variables import backend_access_control_enabled
from cognee.infrastructure.databases.vector import get_vector_engine, get_vectordb_config

logger = logging.getLogger(__name__)

# Assuming your package is named 'cognee' and the migrations are under 'cognee/alembic'
# This is a placeholder for the path logic.
MIGRATIONS_PACKAGE = "cognee"
MIGRATIONS_DIR_NAME = "alembic"


async def run_migrations():
    """
    Finds the Alembic configuration within the installed package and
    programmatically executes 'alembic upgrade head'.
    """
    # 1. Locate the base path of the installed package.
    # This reliably finds the root directory of the installed 'cognee' package.
    # We look for the parent of the 'migrations' directory.
    package_root = str(pkg_resources.files(MIGRATIONS_PACKAGE))

    # 2. Define the paths for config and scripts
    alembic_ini_path = os.path.join(package_root, "alembic.ini")
    script_location_path = os.path.join(package_root, MIGRATIONS_DIR_NAME)

    if not os.path.exists(alembic_ini_path):
        raise FileNotFoundError(
            f"Error: alembic.ini not found at expected locations for package '{MIGRATIONS_PACKAGE}'."
        )
    if not os.path.exists(script_location_path):
        raise FileNotFoundError(
            f"Error: Migrations directory not found at expected locations for package '{MIGRATIONS_PACKAGE}'."
        )

    migration_result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=Path(package_root),
    )

    if migration_result.returncode != 0:

        class MigrationError(Exception):
            """Raised when migrations fail."""

            pass

        migration_output = migration_result.stderr + migration_result.stdout
        logger.error("Migration failed with unexpected error: %s", migration_output)
        raise MigrationError("Relational DB Migrations failed.")

    logger.info("Migration completed successfully.")


async def run_vector_migrations():
    """
    Run the adapter-specific storage migration for ONE vector database: the one
    the current context resolves to — the dataset's own database inside
    ``set_database_global_context_variables``, the global one outside it.

    Pure execution. Per-dataset iteration, context switching, version gating
    and bookkeeping live in :func:`run_pending_vector_migrations`.

    Returns the adapter's migration result, or ``None`` when the adapter has
    no ``run_migrations`` method.
    """
    vector_engine = get_vector_engine()
    migrate_method = getattr(vector_engine, "run_migrations", None)
    if migrate_method is None:
        logger.warning("Vector engine has no run_migrations method. Skipping.")
        return None
    return await migrate_method()


async def run_pending_vector_migrations():
    """
    Version-gating wrapper around :func:`run_vector_migrations`.

    A database whose row already records the current Cognee release was
    migrated by this release before, so it is skipped without resolving its
    vector engine — each release pays the adapter-migration cost once per
    database, not once per startup. For each pending dataset the wrapper enters
    the dataset's database context and runs the migration there, then stamps
    the release on the rows that succeeded; the global row is stamped by
    ``run_database_migrations`` right after in the startup sequence.
    """
    from sqlalchemy import update as sql_update
    from sqlalchemy.exc import OperationalError
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.exceptions import EntityNotFoundError
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.methods.get_dataset_databases import get_dataset_databases
    from cognee.modules.migrations.models import (
        GLOBAL_DATABASE_VERSION_ROW_ID,
        GlobalDatabaseVersion,
    )
    from cognee.modules.users.models import DatasetDatabase
    from cognee.version import get_cognee_version

    current_version = get_cognee_version()
    db_engine = get_relational_engine()

    if not backend_access_control_enabled():
        async with db_engine.get_async_session() as session:
            global_row = await session.get(GlobalDatabaseVersion, GLOBAL_DATABASE_VERSION_ROW_ID)
        if global_row is not None and global_row.cognee_version == current_version:
            return []

        migration_result = await run_vector_migrations()
        if migration_result is None:
            return []
        vector_config = get_vectordb_config()
        logger.info(
            "Vector startup migration completed using provider '%s': %s",
            vector_config.vector_db_provider,
            migration_result,
        )
        return [
            {
                "provider": vector_config.vector_db_provider,
                "vector_database_name": vector_config.vector_db_name,
                "result": migration_result,
            }
        ]

    try:
        dataset_databases = await get_dataset_databases()
    except (OperationalError, EntityNotFoundError) as e:
        logger.warning(
            "Skipping vector startup migrations. Could not access dataset_database table: %s",
            e,
        )
        return []

    pending = [row for row in dataset_databases if row.cognee_version != current_version]
    if len(pending) < len(dataset_databases):
        logger.info(
            "Vector startup migrations skipped for %d dataset(s) already on cognee %s.",
            len(dataset_databases) - len(pending),
            current_version,
        )

    migration_summaries = []
    migrated_dataset_ids = []
    for row in pending:
        try:
            # Resolve the dataset's own vector database the same way every
            # other per-dataset operation does.
            async with set_database_global_context_variables(row.dataset_id, row.owner_id):
                migration_result = await run_vector_migrations()
        except Exception:
            logger.exception(
                "Vector startup migration failed for dataset '%s'; continuing with remaining datasets.",
                row.dataset_id,
            )
            migration_summaries.append({"dataset_id": str(row.dataset_id), "result": "failed"})
            continue
        if migration_result is None:
            continue
        migrated_dataset_ids.append(row.dataset_id)
        migration_summaries.append(
            {
                "dataset_id": str(row.dataset_id),
                "provider": row.vector_database_provider,
                "vector_database_name": row.vector_database_name,
                "result": migration_result,
            }
        )
        logger.info(
            "Vector startup migration completed for dataset '%s' using provider '%s': %s",
            row.dataset_id,
            row.vector_database_provider,
            migration_result,
        )

    # Record the release so these databases are skipped on the next startup.
    if migrated_dataset_ids:
        async with db_engine.get_async_session() as session:
            await session.execute(
                sql_update(DatasetDatabase)
                .where(DatasetDatabase.dataset_id.in_(migrated_dataset_ids))
                .values(cognee_version=current_version)
            )
            await session.commit()

    return migration_summaries


async def run_startup_migrations():
    """
    Run all startup migrations:
    1. relational schema (Alembic) — also creates the version/revision columns
    2. vector schema (adapter-specific; skipped for databases whose row already
       records the current Cognee release)
    3. graph + vector script migrations (revision chain; also records the
       current release on the rows, which is what gates step 2 next startup)
    """
    # Imported lazily to avoid import cycles during ``cognee`` package import.
    from cognee.modules.migrations.runner import run_database_migrations

    await run_migrations()
    await run_pending_vector_migrations()
    await run_database_migrations()
