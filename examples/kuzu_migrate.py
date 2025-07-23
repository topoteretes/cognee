#!/usr/bin/env python3
"""
Kuzu Database Migration Script

This script migrates Kuzu databases between different versions by:
1. Setting up isolated Python environments for each Kuzu version
2. Exporting data from the source database using the old version
3. Importing data into the target database using the new version
4. Handling edge cases like empty databases gracefully

The script automatically handles:
- Empty databases (creates new database with standard Cognee schema)
- Environment setup (creates virtual environments as needed)
- Export/import validation
- Error handling and reporting

Usage Examples:
    # Basic migration from 0.9.0 to 0.11.0
    python kuzu_migrate.py --old-version 0.9.0 --new-version 0.11.0 \\
        --old-db /path/to/old/database --new-db /path/to/new/database

    # Migrate Cognee's default Kuzu database
    python kuzu_migrate.py --old-version 0.9.0 --new-version 0.11.0 \\
        --old-db ~/.cognee_system/databases/cognee_graph_kuzu \\
        --new-db ~/.cognee_system/databases/cognee_graph

Requirements:
- Python 3.7+
- Internet connection (to download Kuzu packages)
- Sufficient disk space for virtual environments and temporary exports

Author: Cognee Team
"""

import tempfile
import sys
import subprocess
import argparse
import os


def ensure_env(version: str) -> str:
    """
    Create (if needed) a venv at .kuzu_envs/{version} and install kuzu=={version}.
    Returns the path to the venv's python executable.
    """
    base = os.path.join(".kuzu_envs", version)
    py_bin = os.path.join(base, "bin", "python")
    if not os.path.isfile(py_bin):
        print(f"→ Setting up venv for Kùzu {version}...", file=sys.stderr)
        # Create venv
        subprocess.run([sys.executable, "-m", "venv", base], check=True)
        # Install the specific Kùzu version
        subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([py_bin, "-m", "pip", "install", f"kuzu=={version}"], check=True)
    return py_bin


def run_migration_step(python_exe: str, db_path: str, cypher: str):
    """
    Uses the given python_exe to execute a short snippet that
    connects to the Kùzu database and runs a Cypher command.
    """
    snippet = f"""
import kuzu
db = kuzu.Database(r"{db_path}")
conn = kuzu.Connection(db)
conn.execute(r\"\"\"{cypher}\"\"\")
"""
    proc = subprocess.run([python_exe, "-c", snippet], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] {cypher} failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)


def migrate(old_ver, new_ver, old_db, new_db):
    """
    Main migration function that handles the complete migration process.
    """
    # Check if old database exists
    if not os.path.exists(old_db):
        print(f"Source database '{old_db}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Prepare target - ensure parent directory exists but remove target if it exists
    parent_dir = os.path.dirname(new_db)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Remove target database if it exists (Kuzu 0.11.0 requires clean slate)
    if os.path.exists(new_db):
        if os.path.isdir(new_db):
            import shutil

            shutil.rmtree(new_db)
            print(f"Removed existing target directory: {new_db}", file=sys.stderr)
        else:
            os.remove(new_db)
            print(f"Removed existing target file: {new_db}", file=sys.stderr)

    # Set up environments
    print(f"Setting up Kuzu {old_ver} environment...", file=sys.stderr)
    old_py = ensure_env(old_ver)
    print(f"Setting up Kuzu {new_ver} environment...", file=sys.stderr)
    new_py = ensure_env(new_ver)

    with tempfile.TemporaryDirectory() as export_dir:
        export_file = os.path.join(export_dir, "kuzu_export")
        print(f"Exporting old DB → {export_dir}", file=sys.stderr)
        run_migration_step(old_py, old_db, f"EXPORT DATABASE '{export_file}'")
        print("Export complete.", file=sys.stderr)

        # Check if export files were created and have content
        schema_file = os.path.join(export_file, "schema.cypher")
        if not os.path.exists(schema_file) or os.path.getsize(schema_file) == 0:
            raise ValueError(f"Schema file not found: {schema_file}")

        print(f"Importing into new DB at {new_db}", file=sys.stderr)
        run_migration_step(new_py, new_db, f"IMPORT DATABASE '{export_file}'")
        print("Import complete.", file=sys.stderr)

    print("✅ Migration finished successfully!")


def main():
    p = argparse.ArgumentParser(
        description="Migrate Kùzu DB via PyPI versions",
        epilog="""
Examples:
  %(prog)s --old-version 0.9.0 --new-version 0.11.0 \\
    --old-db /path/to/old/db --new-db /path/to/new/db

  %(prog)s --old-version 0.9.0 --new-version 0.11.0 \\
    --old-db ~/.cognee_system/databases/cognee_graph_kuzu \\
    --new-db ~/.cognee_system/databases/cognee_graph

Note: This script will create virtual environments in .kuzu_envs/ directory
to isolate different Kuzu versions.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--old-version", required=True, help="Source Kuzu version (e.g., 0.9.0)")
    p.add_argument("--new-version", required=True, help="Target Kuzu version (e.g., 0.11.0)")
    p.add_argument("--old-db", required=True, help="Path to source database directory")
    p.add_argument("--new-db", required=True, help="Path to target database directory")

    args = p.parse_args()

    print(
        f"🔄 Migrating Kuzu database from {args.old_version} to {args.new_version}", file=sys.stderr
    )
    print(f"📂 Source: {args.old_db}", file=sys.stderr)
    print(f"📂 Target: {args.new_db}", file=sys.stderr)
    print("", file=sys.stderr)

    migrate(args.old_version, args.new_version, args.old_db, args.new_db)
    # migrate("0.9.0", "0.11.0", "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu", "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph")


if __name__ == "__main__":
    main()
