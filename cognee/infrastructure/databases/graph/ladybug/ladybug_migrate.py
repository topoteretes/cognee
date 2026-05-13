#!/usr/bin/env python3
"""
Ladybug Database Migration Script

This script migrates legacy Kuzu/Ladybug databases between different versions by:
1. Setting up isolated Python environments for each database package version
2. Exporting data from the source database using the old version
3. Importing data into the target database using the new version
4. If overwrite is enabled target database will replace source database and source database will have the prefix _old
5. If delete-old is enabled target database will be renamed to source database and source database will be deleted

The script automatically handles:
- Environment setup (creates virtual environments as needed)
- Export/import validation
- Error handling and reporting

Usage Examples:
    # Basic migration from Kuzu 0.9.0 to Ladybug 0.16.0
    python ladybug_migrate.py --old-version 0.9.0 --new-version 0.16.0 --old-db /path/to/old/database --new-db /path/to/new/database

Requirements:
- Python 3.7+
- Internet connection (to download Kuzu/Ladybug packages)
- Sufficient disk space for virtual environments and temporary exports

Notes:
- Legacy Kuzu package versions are used for old databases that predate the Ladybug rename.
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
    39: "0.11.3",
}


def read_ladybug_storage_version(ladybug_db_path: str) -> str:
    """
    Reads the storage version code from the first catalog file bytes.

    :param ladybug_db_path: Path to the Ladybug database file/directory.
    :return: Storage version as a version string.
    """
    if os.path.isdir(ladybug_db_path):
        version_file_path = os.path.join(ladybug_db_path, "catalog.kz")
        if not os.path.isfile(version_file_path):
            raise FileNotFoundError("Ladybug catalog.kz file does not exist")
    else:
        version_file_path = ladybug_db_path

    with open(version_file_path, "rb") as f:
        # Skip the 3-byte magic "KUZ" and one byte of padding
        f.seek(4)
        # Read the next 8 bytes as a little-endian unsigned 64-bit integer
        data = f.read(8)
        if len(data) < 8:
            raise ValueError(f"File '{version_file_path}' does not contain a storage version code.")
        version_code = struct.unpack("<Q", data)[0]

    if ladybug_version_mapping.get(version_code):
        return ladybug_version_mapping[version_code]
    else:
        raise ValueError("Could not map version_code to proper Ladybug version.")


def try_read_ladybug_storage_version(ladybug_db_path: str):
    """Best-effort variant of ``read_ladybug_storage_version``.

    Returns the mapped version string if the catalog file exists and its
    version_code is present in ``ladybug_version_mapping``. Returns ``None``
    when:

      * the catalog file is missing (truly fresh path), or
      * the version_code is newer than anything in the mapping (typically
        a fresh DB written by a ladybug release the mapping table doesn't
        list yet — at the time of writing the table tops out at 0.11.3,
        so any code emitted by ladybug >= 0.12 lands here).

    This is the lookup the runtime bootstrap path wants — the legacy
    migration script keeps using ``read_ladybug_storage_version`` so its
    CLI surface still raises on bad input.
    """
    try:
        return read_ladybug_storage_version(ladybug_db_path)
    except (FileNotFoundError, ValueError):
        return None


def _package_for_version(version: str) -> tuple[str, str]:
    version_parts = tuple(int(part) for part in version.split(".")[:3])
    package_name = "ladybug" if version_parts >= (0, 15, 0) else "kuzu"
    module_name = "ladybug" if package_name == "ladybug" else "kuzu"
    return package_name, module_name


def ensure_env(version: str, export_dir) -> tuple[str, str]:
    """
    Create a venv for a package version and return its python executable and module name.
    """
    # Use temp directory to create venv
    package_name, module_name = _package_for_version(version)
    envs_dir = os.path.join(export_dir, ".ladybug_envs")

    # venv base under the script directory
    base = os.path.join(envs_dir, f"{package_name}_{version}")
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    python_executable = "python.exe" if os.name == "nt" else "python"
    py_bin = os.path.join(base, scripts_dir, python_executable)
    # If environment already exists clean it
    if os.path.isfile(py_bin):
        shutil.rmtree(base)

    print(f"Setting up venv for {package_name} {version}...", file=sys.stderr)
    # Create venv
    # NOTE: Running python in debug mode can cause issues with creating a virtual environment from that python instance
    subprocess.run([sys.executable, "-m", "venv", base], check=True)
    # Install the specific package version
    subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([py_bin, "-m", "pip", "install", f"{package_name}=={version}"], check=True)
    return py_bin, module_name


def run_migration_step(python_exe: str, module_name: str, db_path: str, cypher: str):
    """
    Uses the given python_exe to execute a short snippet that
    connects to the database and runs a Cypher command.
    """
    db_path = os.path.abspath(db_path)
    snippet = f"""
import {module_name} as graph_db
db = graph_db.Database({db_path!r})
conn = graph_db.Connection(db)
conn.execute({cypher!r})
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [python_exe, "-c", snippet],
        capture_output=True,
        text=True,
        cwd=tempfile.gettempdir(),
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{cypher} failed:\n{proc.stderr}")


def ladybug_migration(
    new_db, old_db, new_version, old_version=None, overwrite=None, delete_old=None
):
    """
    Main migration function that handles the complete migration process.
    """
    print(f"Migrating graph database from {old_version} to {new_version}", file=sys.stderr)
    print(f"Source: {old_db}", file=sys.stderr)
    print("", file=sys.stderr)

    # If version of old database is not provided try to determine it based on file info
    if not old_version:
        old_version = read_ladybug_storage_version(old_db)

    # Check if old database exists
    if not os.path.exists(old_db):
        raise FileNotFoundError(f"Source database '{old_db}' does not exist.")

    # Prepare target - ensure parent directory exists but remove target if it exists
    parent_dir = os.path.dirname(new_db)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(new_db):
        raise FileExistsError(
            "File already exists at new database location, remove file or change new database file path to continue"
        )

    # Use temp directory for all processing, it will be cleaned up after with statement
    with tempfile.TemporaryDirectory() as export_dir:
        # Set up environments
        print(f"Setting up graph database {old_version} environment...", file=sys.stderr)
        old_py, old_module = ensure_env(old_version, export_dir)
        print(f"Setting up graph database {new_version} environment...", file=sys.stderr)
        new_py, new_module = ensure_env(new_version, export_dir)

        export_file = os.path.join(export_dir, "ladybug_export")
        print(f"Exporting old DB to {export_dir}", file=sys.stderr)
        run_migration_step(old_py, old_module, old_db, f"EXPORT DATABASE '{export_file}'")
        print("Export complete.", file=sys.stderr)

        # Check if export files were created and have content
        schema_file = os.path.join(export_file, "schema.cypher")
        if not os.path.exists(schema_file) or os.path.getsize(schema_file) == 0:
            raise ValueError(f"Schema file not found: {schema_file}")

        print(f"Importing into new DB at {new_db}", file=sys.stderr)
        run_migration_step(new_py, new_module, new_db, f"IMPORT DATABASE '{export_file}'")
        print("Import complete.", file=sys.stderr)

    # Rename new database to old database name if enabled
    if overwrite or delete_old:
        # Remove stale locks before replacing the original database path.
        for lock_file in (new_db + ".lock", old_db + ".lock"):
            if os.path.exists(lock_file):
                os.remove(lock_file)
        rename_databases(old_db, old_version, new_db, delete_old)

    print("Ladybug graph database migration finished successfully!")


def rename_databases(old_db: str, old_version: str, new_db: str, delete_old: bool):
    """
    When overwrite is enabled, back up the original old_db (file with .lock and .wal or directory)
    by renaming it to *_old, and replace it with the newly imported new_db files.

    When delete_old is enabled replace the old database with the new one and delete old database
    """
    base_dir = os.path.dirname(old_db)
    name = os.path.basename(old_db.rstrip(os.sep))
    # Add _old_ and version info to backup graph database
    backup_database_name = f"{name}_old_" + old_version.replace(".", "_")
    backup_base = os.path.join(base_dir, backup_database_name)

    if os.path.isfile(old_db):
        # File-based database: handle main file and accompanying lock/WAL
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
        # Directory-based Ladybug database
        backup_dir = backup_base
        if delete_old:
            shutil.rmtree(old_db)
        else:
            os.rename(old_db, backup_dir)
            print(f"Renamed directory '{old_db}' to '{backup_dir}'", file=sys.stderr)
    else:
        raise FileNotFoundError(f"Original database path '{old_db}' not found for renaming.")

    # Now move new files into place
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
  %(prog)s --old-version 0.9.0 --new-version 0.16.0 \\
    --old-db /path/to/old/db --new-db /path/to/new/db --overwrite

Note: This script will create temporary virtual environments in .ladybug_envs/ directory
to isolate different database package versions.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--old-version",
        required=False,
        default=None,
        help="Source database version (e.g., 0.9.0). If not provided automatic version detection will be attempted.",
    )
    p.add_argument("--new-version", required=True, help="Target Ladybug version (e.g., 0.16.0)")
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
