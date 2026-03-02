#!/usr/bin/env python3
"""
Ladybug Database Migration Script

This script migrates Ladybug databases between different versions by:
1. Setting up isolated Python environments for each Ladybug version
2. Exporting data from the source database using the old version
3. Importing data into the target database using the new version
4. If overwrite is enabled target database will replace source database and source database will have the prefix _old
5. If delete-old is enabled target database will be renamed to source database and source database will be deleted

The script automatically handles:
- Environment setup (creates virtual environments as needed)
- Export/import validation
- Error handling and reporting

Usage Examples:
    # Basic migration from 0.9.0 to latest
    python ladybug_migrate.py --old-version 0.9.0 --new-version latest --old-db /path/to/old/database --new-db /path/to/new/database

Requirements:
- Python 3.7+
- Internet connection (to download Ladybug packages)
- Sufficient disk space for virtual environments and temporary exports

Notes:
- Can be used to migrate from Kuzu or Ladybug databases
- Ladybug is built on top of Kuzu, so migration from Kuzu should work seamlessly
"""

import tempfile
import sys
import struct
import shutil
import subprocess
import argparse
import os


ladybug_version_mapping = {
    34: "0.7.0",
    35: "0.7.1",
    36: "0.8.2",
    37: "0.9.0",
    38: "0.10.1",
    39: "0.11.0",
}


def read_ladybug_storage_version(ladybug_db_path: str) -> str:
    """
    Reads the Ladybug/Kuzu storage version code from the first catalog.bin file bytes.

    :param ladybug_db_path: Path to the Ladybug database file/directory.
    :return: Storage version code as a string.
    """
    if os.path.isdir(ladybug_db_path):
        version_file_path = os.path.join(ladybug_db_path, "catalog.kz")
        if os.path.isfile(version_file_path):
            catalog_path = version_file_path
        else:
            catalog_path = os.path.join(ladybug_db_path, "catalog.lbug")
    else:
        catalog_path = ladybug_db_path

    if not os.path.isfile(catalog_path):
        raise FileExistsError(f"Catalog file does not exist at {catalog_path}")

    with open(catalog_path, "rb") as f:
        f.seek(4)
        data = f.read(8)
        if len(data) < 8:
            raise ValueError(f"File '{catalog_path}' does not contain a storage version code.")
        version_code = struct.unpack("<Q", data)[0]

    if ladybug_version_mapping.get(version_code):
        return ladybug_version_mapping[version_code]
    else:
        raise ValueError("Could not map version_code to proper Ladybug version.")


def ensure_env(version: str, export_dir: str) -> str:
    """
    Create (if needed) a venv at .ladybug_envs/{version} and install real_ladybug=={version}.
    Returns the path to the venv's python executable.
    """
    ladybug_envs_dir = os.path.join(export_dir, ".ladybug_envs")

    base = os.path.join(ladybug_envs_dir, version)
    py_bin = os.path.join(base, "bin", "python")
    if os.path.isfile(py_bin):
        shutil.rmtree(base)

    print(f"→ Setting up venv for Ladybug {version}...", file=sys.stderr)
    subprocess.run([sys.executable, "-m", "venv", base], check=True)
    subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if version == "latest":
        subprocess.run([py_bin, "-m", "pip", "install", "real_ladybug"], check=True)
    else:
        subprocess.run([py_bin, "-m", "pip", "install", f"real_ladybug=={version}"], check=True)
    return py_bin


def run_migration_step(python_exe: str, db_path: str, cypher: str):
    """
    Uses the given python_exe to execute a short snippet that
    connects to the Ladybug database and runs a Cypher command.
    """
    snippet = f"""
import real_ladybug as lb
db = lb.Database(r"{db_path}")
conn = lb.Connection(db)
conn.execute(r\"\"\"{cypher}\"\"\")
"""
    proc = subprocess.run([python_exe, "-c", snippet], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] {cypher} failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(proc.returncode)


def ladybug_migration(
    new_db: str,
    old_db: str,
    new_version: str,
    old_version: str = None,
    overwrite: bool = None,
    delete_old: bool = None,
):
    """
    Main migration function that handles the complete migration process.
    """
    print(f"🔄 Migrating Ladybug database from {old_version} to {new_version}", file=sys.stderr)
    print(f"📂 Source: {old_db}", file=sys.stderr)
    print("", file=sys.stderr)

    if not old_version:
        old_version = read_ladybug_storage_version(old_db)

    if not os.path.exists(old_db):
        print(f"Source database '{old_db}' does not exist.", file=sys.stderr)
        sys.exit(1)

    parent_dir = os.path.dirname(new_db)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(new_db):
        raise FileExistsError(
            "File already exists at new database location, remove file or change new database file path to continue"
        )

    with tempfile.TemporaryDirectory() as export_dir:
        print(f"Setting up Ladybug {old_version} environment...", file=sys.stderr)
        old_py = ensure_env(old_version, export_dir)
        print(f"Setting up Ladybug {new_version} environment...", file=sys.stderr)
        new_py = ensure_env(new_version, export_dir)

        export_file = os.path.join(export_dir, "ladybug_export")
        print(f"Exporting old DB → {export_dir}", file=sys.stderr)
        run_migration_step(old_py, old_db, f"EXPORT DATABASE '{export_file}'")
        print("Export complete.", file=sys.stderr)

        schema_file = os.path.join(export_file, "schema.cypher")
        if not os.path.exists(schema_file) or os.path.getsize(schema_file) == 0:
            raise ValueError(f"Schema file not found: {schema_file}")

        print(f"Importing into new DB at {new_db}", file=sys.stderr)
        run_migration_step(new_py, new_db, f"IMPORT DATABASE '{export_file}'")
        print("Import complete.", file=sys.stderr)

    if overwrite or delete_old:
        lock_file = new_db + ".lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)
        rename_databases(old_db, old_version, new_db, delete_old)

    print("✅ Ladybug graph database migration finished successfully!")


def rename_databases(old_db: str, old_version: str, new_db: str, delete_old: bool):
    """
    When overwrite is enabled, back up the original old_db by renaming it to *_old,
    and replace it with the newly imported new_db files.

    When delete_old is enabled replace the old database with the new one and delete old database
    """
    base_dir = os.path.dirname(old_db)
    name = os.path.basename(old_db.rstrip(os.sep))
    backup_database_name = f"{name}_old_" + old_version.replace(".", "_")
    backup_base = os.path.join(base_dir, backup_database_name)

    if os.path.isfile(old_db):
        for ext in ["", ".wal"]:
            src = old_db + ext
            dst = backup_base + ext
            if os.path.exists(src):
                if delete_old:
                    os.remove(src)
                else:
                    os.rename(src, dst)
                    print(f"Renamed '{src}' to '{dst}'", file=sys.stderr)
    elif os.path.isdir(old_db):
        backup_dir = backup_base
        if delete_old:
            shutil.rmtree(old_db)
        else:
            os.rename(old_db, backup_dir)
            print(f"Renamed directory '{old_db}' to '{backup_dir}'", file=sys.stderr)
    else:
        print(f"Original database path '{old_db}' not found for renaming.", file=sys.stderr)
        sys.exit(1)

    for ext in ["", ".wal"]:
        src_new = new_db + ext
        dst_new = os.path.join(base_dir, name + ext)
        if os.path.exists(src_new):
            os.rename(src_new, dst_new)
            print(f"Renamed '{src_new}' to '{dst_new}'", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(
        description="Migrate Ladybug DB via PyPI versions",
        epilog="""
Examples:
  %(prog)s --old-version 0.9.0 --new-version latest \\
    --old-db /path/to/old/db --new-db /path/to/new/db --overwrite

Note: This script will create temporary virtual environments in .ladybug_envs/ directory
to isolate different Ladybug versions.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--old-version",
        required=False,
        default=None,
        help="Source Ladybug version (e.g., 0.9.0). If not provided automatic version detection will be attempted.",
    )
    p.add_argument("--new-version", required=True, help="Target Ladybug version (e.g., latest)")
    p.add_argument("--old-db", required=True, help="Path to source database directory")
    p.add_argument(
        "--new-db",
        required=True,
        help="Path to target database directory, it can't be the same path as the old database. Use the overwrite flag if you want to replace the old database with the new one.",
    )
    p.add_argument(
        "--overwrite",
        required=False,
        action="store_true",
        default=False,
        help="Rename new-db to the old-db name and location, keeps old-db as backup if delete-old is not True",
    )
    p.add_argument(
        "--delete-old",
        required=False,
        action="store_true",
        default=False,
        help="When overwrite and delete-old is True old-db will not be stored as backup",
    )

    args = p.parse_args()

    ladybug_migration(
        new_db=args.new_db,
        old_db=args.old_db,
        new_version=args.new_version,
        old_version=args.old_version,
        overwrite=args.overwrite,
        delete_old=args.delete_old,
    )


if __name__ == "__main__":
    main()
