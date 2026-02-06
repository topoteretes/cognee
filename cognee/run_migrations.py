import os
import sys
import subprocess
from pathlib import Path
import importlib.resources as pkg_resources

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
        ["python", "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
        cwd=Path(package_root),
    )

    if migration_result.returncode != 0:
        migration_output = migration_result.stderr + migration_result.stdout
        print(f"Migration failed with unexpected error: {migration_output}")
        sys.exit(1)

    print("Migration completed successfully.")
