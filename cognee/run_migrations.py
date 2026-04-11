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
    from cognee.infrastructure.databases.vector import get_vector_engine

    vector_engine = get_vector_engine()
    migrate_method = getattr(vector_engine, "run_migrations", None)
    if migrate_method is None:
        logger.warning("Vector engine has no run_migrations method. Skipping.")
        return

    migration_result = await migrate_method()
    logger.info(
        "Vector startup migration completed for provider '%s': %s",
        getattr(vector_engine, "name", "unknown"),
        migration_result,
    )


async def run_startup_migrations():
    """
    Run all startup migrations:
    1. relational schema (Alembic)
    2. vector schema (adapter-specific)
    """
    await run_migrations()
    await run_vector_migrations()
