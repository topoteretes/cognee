#!/usr/bin/env python3
"""
Kuzu Database Migration Script

This script migrates Kuzu databases between different versions by:
1. Setting up isolated Python environments for each Kuzu version
2. Exporting data from the source database using the old version
3. Importing data into the target database using the new version
4. If overwrite is enabled target database will replace source database and source database will have the prefix _old
5. If delete-old is enabled target database will be renamed to source database and source database will be deleted

The script automatically handles:
- Environment setup (creates virtual environments as needed)
- Export/import validation
- Error handling and reporting

Usage Examples:
    # Basic migration from 0.9.0 to 0.11.0
    python kuzu_migrate.py --old-version 0.9.0 --new-version 0.11.0 --old-db /path/to/old/database --new-db /path/to/new/database

Requirements:
- Python 3.7+
- Internet connection (to download Kuzu packages)
- Sufficient disk space for virtual environments and temporary exports

Notes:
- Can only be used to migrate to newer Kuzu versions, from 0.11.0 onwards
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
    # directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # venv base under the script directory
    base = os.path.join(script_dir, ".kuzu_envs", version)
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

    if os.path.exists(new_db):
        raise FileExistsError(
            "File already exists at new database location, remove file or change new database file path to continue"
        )

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


def rename_databases(old_db: str, new_db: str, delete_old: bool):
    """
    When overwrite is enabled, back up the original old_db (file with .lock and .wal or directory)
    by renaming it to *_old, and replace it with the newly imported new_db files.

    When delete_old is enabled replace the old database with the new one and delete old database
    """
    base_dir = os.path.dirname(old_db)
    name = os.path.basename(old_db.rstrip(os.sep))
    backup_base = os.path.join(base_dir, f"{name}_old")

    if os.path.isfile(old_db):
        # File-based database: handle main file and accompanying lock/WAL
        for ext in ["", ".lock", ".wal"]:
            src = old_db + ext
            dst = backup_base + ext
            if os.path.exists(src):
                if delete_old:
                    os.remove(src)
                else:
                    os.rename(src, dst)
                    print(f"Renamed '{src}' to '{dst}'", file=sys.stderr)
    elif os.path.isdir(old_db):
        # Directory-based Kuzu database
        backup_dir = backup_base
        if delete_old:
            import shutil

            shutil.rmtree(old_db)
        else:
            os.rename(old_db, backup_dir)
            print(f"Renamed directory '{old_db}' to '{backup_dir}'", file=sys.stderr)
    else:
        print(f"Original database path '{old_db}' not found for renaming.", file=sys.stderr)
        sys.exit(1)

    # Now move new files into place
    for ext in ["", ".lock", ".wal"]:
        src_new = new_db + ext
        dst_new = os.path.join(base_dir, name + ext)
        if os.path.exists(src_new):
            os.rename(src_new, dst_new)
            print(f"Renamed '{src_new}' to '{dst_new}'", file=sys.stderr)


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
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Rename new-db to the old-db name and location, keeps old-db as backup if delete-old is not True",
    )
    p.add_argument(
        "--delete-old",
        action="store_true",
        default=False,
        help="When overwrite and delete-old is True old-db will not be stored as backup",
    )

    args = p.parse_args()

    print(
        f"🔄 Migrating Kuzu database from {args.old_version} to {args.new_version}", file=sys.stderr
    )
    print(f"📂 Source: {args.old_db}", file=sys.stderr)
    print(f"📂 Target: {args.new_db}", file=sys.stderr)
    print("", file=sys.stderr)

    migrate(args.old_version, args.new_version, args.old_db, args.new_db)

    if args.overwrite or args.delete_old:
        rename_databases(args.old_db, args.new_db, args.delete_old)

    # migrate("0.9.0", "0.11.0", "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu", "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph")
    # rename_databases("/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph_kuzu", "/Users/igorilic/Desktop/cognee/cognee/.cognee_system/databases/cognee_graph")


if __name__ == "__main__":
    main()
