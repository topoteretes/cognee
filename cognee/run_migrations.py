import os
import sys
import logging
import subprocess
from pathlib import Path
import importlib.resources as pkg_resources

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
    Run adapter-specific vector storage migrations at startup.
    """
    from sqlalchemy.exc import OperationalError
    from sqlalchemy import select
    from cognee.infrastructure.databases.exceptions import EntityNotFoundError
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector.create_vector_engine import create_vector_engine
    from cognee.infrastructure.databases.utils.resolve_dataset_database_connection_info import (
        resolve_dataset_database_connection_info,
    )
    from cognee.modules.users.models import DatasetDatabase

    db_engine = get_relational_engine()
    try:
        async with db_engine.get_async_session() as session:
            dataset_databases = (await session.scalars(select(DatasetDatabase))).all()
    except (OperationalError, EntityNotFoundError) as e:
        logger.debug(
            "Skipping vector startup migrations. Could not access dataset_database table: %s",
            e,
        )
        return []

    migration_summaries = []
    for dataset_database in dataset_databases:
        dataset_database = await resolve_dataset_database_connection_info(dataset_database)

        connection_info = getattr(dataset_database, "vector_database_connection_info", {}) or {}
        vector_engine = create_vector_engine(
            vector_db_provider=dataset_database.vector_database_provider,
            vector_db_url=dataset_database.vector_database_url,
            vector_db_name=dataset_database.vector_database_name,
            vector_db_port=str(connection_info.get("port", "") or ""),
            vector_db_key=dataset_database.vector_database_key or "",
            vector_dataset_database_handler=dataset_database.vector_dataset_database_handler or "",
            vector_db_username=connection_info.get("username", "") or "",
            vector_db_password=connection_info.get("password", "") or "",
            vector_db_host=connection_info.get("host", "") or "",
        )

        migrate_method = getattr(vector_engine, "run_migrations", None)
        if migrate_method is None:
            logger.warning(
                "Vector engine has no run_migrations method for dataset '%s'. Skipping.",
                dataset_database.dataset_id,
            )
            continue

        migration_result = await migrate_method()
        summary = {
            "dataset_id": str(dataset_database.dataset_id),
            "provider": dataset_database.vector_database_provider,
            "vector_database_name": dataset_database.vector_database_name,
            "result": migration_result,
        }
        migration_summaries.append(summary)
        logger.info(
            "Vector startup migration completed for dataset '%s' using provider '%s': %s",
            dataset_database.dataset_id,
            dataset_database.vector_database_provider,
            migration_result,
        )

    return migration_summaries


async def run_startup_migrations():
    """
    Run all startup migrations:
    1. relational schema (Alembic)
    2. vector schema (adapter-specific)
    """
    await run_migrations()
    await run_vector_migrations()
