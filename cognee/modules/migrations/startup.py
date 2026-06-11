"""Startup migration orchestration.

Two stages, in order:

1. relational schema (Alembic) — must run first: it creates the revision
   columns / tables the next stage reads.
2. graph + vector script migrations (revision chains, ``runner.py``) — this
   includes the vector adapter's own storage-schema migration, which is a
   chain entry (``adapter_storage_migration``), so it is gated, locked and
   failure-isolated exactly like every other migration.

Triggered from the FastAPI lifespan on every server start, from ``remember()``'s
first call in an SDK process, and explicitly via ``cognee.run_startup_migrations()``.
"""

import logging
import os
import sys
import subprocess
from pathlib import Path
import importlib.resources as pkg_resources

logger = logging.getLogger(__name__)

MIGRATIONS_PACKAGE = "cognee"
MIGRATIONS_DIR_NAME = "alembic"


class MigrationError(Exception):
    """Raised when migrations fail."""


async def run_relational_migrations():
    """
    Finds the Alembic configuration within the installed package and
    programmatically executes 'alembic upgrade head'.
    """
    # 1. Locate the base path of the installed package.
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
        migration_output = migration_result.stderr + migration_result.stdout
        logger.error("Migration failed with unexpected error: %s", migration_output)
        raise MigrationError("Relational DB Migrations failed.")

    logger.info("Migration completed successfully.")


async def run_startup_migrations():
    """Run all startup migrations: relational schema first, then the graph +
    vector revision chains (see module docstring)."""
    from cognee.modules.migrations.runner import run_database_migrations

    await run_relational_migrations()
    await run_database_migrations()
