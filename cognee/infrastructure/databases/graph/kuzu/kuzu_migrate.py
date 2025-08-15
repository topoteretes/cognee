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

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any, Optional


# Lazy-import s3fs via our storage adapter only when needed, so local-only runs
# don't require S3 credentials or dependencies at import time.
def _is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def _get_s3_client() -> Any:  # Returns configured s3fs client via project storage adapter
    from cognee.infrastructure.files.storage.S3FileStorage import S3FileStorage

    storage: Any = S3FileStorage("")
    client: Any = storage.s3  # type: ignore[attr-defined]
    return client


kuzu_version_mapping = {
    34: "0.7.0",
    35: "0.7.1",
    36: "0.8.2",
    37: "0.9.0",
    38: "0.10.1",
    39: "0.11.0",
}


def read_kuzu_storage_version(kuzu_db_path: str) -> str:
    """
    Read the Kuzu storage version from the first bytes of catalog.kz and map it
    to a human-readable Kuzu semantic version string (e.g. "0.9.0").

    :param kuzu_db_path: Path/URI (local or s3://) to the Kuzu database file/directory.
    :return: Semantic version string (e.g. "0.9.0").
    """
    if _is_s3_path(kuzu_db_path):
        s3 = _get_s3_client()
        # Determine whether the remote path is a directory or file
        version_key = kuzu_db_path
        try:
            if s3.isdir(kuzu_db_path):
                version_key = kuzu_db_path.rstrip("/") + "/catalog.kz"
            # Open directly from S3 without downloading the entire DB
            with s3.open(version_key, "rb") as f:
                f.seek(4)
                data = f.read(8)
        except FileNotFoundError:
            raise FileExistsError("Kuzu catalog.kz file does not exist on S3")
    else:
        if os.path.isdir(kuzu_db_path):
            kuzu_version_file_path = os.path.join(kuzu_db_path, "catalog.kz")
            if not os.path.isfile(kuzu_version_file_path):
                raise FileExistsError("Kuzu catalog.kz file does not exist")
        else:
            kuzu_version_file_path = kuzu_db_path

        with open(kuzu_version_file_path, "rb") as f:
            # Skip the 3-byte magic "KUZ" and one byte of padding
            f.seek(4)
            # Read the next 8 bytes as a little-endian unsigned 64-bit integer
            data = f.read(8)
            if len(data) < 8:
                raise ValueError(
                    f"File '{kuzu_version_file_path}' does not contain a storage version code."
                )

    if len(data) < 8:
        raise ValueError("catalog.kz does not contain a storage version code.")
    version_code = struct.unpack("<Q", data)[0]

    if kuzu_version_mapping.get(version_code):
        return kuzu_version_mapping[version_code]
    else:
        raise ValueError("Could not map version_code to proper Kuzu version.")


def ensure_env(version: str, export_dir: str) -> str:
    """
    Create (if needed) a venv at .kuzu_envs/{version} and install kuzu=={version}.
    Returns the path to the venv's python executable.
    """
    # Use temp directory to create venv
    kuzu_envs_dir = os.path.join(export_dir, ".kuzu_envs")

    # venv base under the script directory
    base = os.path.join(kuzu_envs_dir, version)
    py_bin = os.path.join(base, "bin", "python")
    # If environment already exists clean it
    if os.path.isfile(py_bin):
        shutil.rmtree(base)

    print(f"→ Setting up venv for Kùzu {version}...", file=sys.stderr)
    # Create venv
    # NOTE: Running python in debug mode can cause issues with creating a virtual environment from that python instance
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


def kuzu_migration(
    new_db: str,
    old_db: str,
    new_version: str,
    old_version: Optional[str] = None,
    overwrite: Optional[bool] = None,
    delete_old: Optional[bool] = None,
) -> None:
    """
    Main migration function that handles the complete migration process.
    """
    print(f"🔄 Migrating Kuzu database from {old_version} to {new_version}", file=sys.stderr)
    print(f"📂 Source: {old_db}", file=sys.stderr)
    print("", file=sys.stderr)

    # If version of old kuzu db is not provided try to determine it based on file info
    if not old_version:
        old_version = read_kuzu_storage_version(old_db)

    # Check if old database exists (local or S3)
    if _is_s3_path(old_db):
        s3 = _get_s3_client()
        if not (s3.exists(old_db) or s3.exists(old_db.rstrip("/") + "/")):
            print(f"Source database '{old_db}' does not exist.", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.path.exists(old_db):
            print(f"Source database '{old_db}' does not exist.", file=sys.stderr)
            sys.exit(1)

    # Prepare target - ensure parent directory exists but remove target if it exists
    parent_dir = os.path.dirname(new_db)
    if _is_s3_path(new_db):
        # For S3 we don't create directories locally; just ensure the key doesn't already exist
        s3 = _get_s3_client()
        if s3.exists(new_db) or s3.exists(new_db.rstrip("/") + "/"):
            raise FileExistsError(
                "File already exists at new database location on S3; remove it or change new database path to continue"
            )
    else:
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        if os.path.exists(new_db):
            raise FileExistsError(
                "File already exists at new database location, remove file or change new database file path to continue"
            )

    # Use temp directory for all processing, it will be cleaned up after with statement
    with tempfile.TemporaryDirectory() as export_dir:
        is_old_s3 = _is_s3_path(old_db)
        is_new_s3 = _is_s3_path(new_db)

        # If old DB is on S3, download it locally first.
        local_old_db = old_db
        local_new_db = new_db
        if is_old_s3:
            s3 = _get_s3_client()
            local_old_db = os.path.join(export_dir, "old_kuzu_db")
            # Download either a file or a directory recursively
            print(f"⬇️  Downloading old DB from S3 → {local_old_db}", file=sys.stderr)
            s3.get(old_db, local_old_db, recursive=True)

        if is_new_s3:
            # Always stage new DB locally, then upload after migration
            local_new_db = os.path.join(export_dir, "new_kuzu_db")
        # Set up environments
        print(f"Setting up Kuzu {old_version} environment...", file=sys.stderr)
        old_py = ensure_env(old_version, export_dir)
        print(f"Setting up Kuzu {new_version} environment...", file=sys.stderr)
        new_py = ensure_env(new_version, export_dir)

        export_file = os.path.join(export_dir, "kuzu_export")
        print(f"Exporting old DB → {export_dir}", file=sys.stderr)
        run_migration_step(old_py, local_old_db, f"EXPORT DATABASE '{export_file}'")
        print("Export complete.", file=sys.stderr)

        # Check if export files were created and have content
        schema_file = os.path.join(export_file, "schema.cypher")
        if not os.path.exists(schema_file) or os.path.getsize(schema_file) == 0:
            raise ValueError(f"Schema file not found: {schema_file}")

        print(f"Importing into new DB at {local_new_db}", file=sys.stderr)
        run_migration_step(new_py, local_new_db, f"IMPORT DATABASE '{export_file}'")
        print("Import complete.", file=sys.stderr)

        # If the target is S3, upload the migrated DB now
        if is_new_s3:
            # Remove kuzu lock from migrated DB before upload if present
            lock_file = local_new_db + ".lock"
            if os.path.exists(lock_file):
                os.remove(lock_file)

            print(f"⬆️  Uploading new DB to S3: {new_db}", file=sys.stderr)
            s3 = _get_s3_client()
            s3.put(local_new_db, new_db, recursive=True)

    # Normalize flags
    overwrite = bool(overwrite)
    delete_old = bool(delete_old)

    # Rename/move results into place if requested
    if overwrite or delete_old:
        if _is_s3_path(new_db) or _is_s3_path(old_db):
            # S3-aware rename
            _s3_rename_databases(old_db, old_version, new_db, delete_old)
        else:
            # Remove kuzu lock from migrated DB
            lock_file = new_db + ".lock"
            if os.path.exists(lock_file):
                os.remove(lock_file)
            rename_databases(old_db, old_version, new_db, delete_old)

    print("✅ Kuzu graph database migration finished successfully!")


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
        # Directory-based Kuzu database
        backup_dir = backup_base
        if delete_old:
            shutil.rmtree(old_db)
        else:
            os.rename(old_db, backup_dir)
            print(f"Renamed directory '{old_db}' to '{backup_dir}'", file=sys.stderr)
    else:
        print(f"Original database path '{old_db}' not found for renaming.", file=sys.stderr)
        sys.exit(1)

    # Now move new files into place
    for ext in ["", ".wal"]:
        src_new = new_db + ext
        dst_new = os.path.join(base_dir, name + ext)
        if os.path.exists(src_new):
            os.rename(src_new, dst_new)
            print(f"Renamed '{src_new}' to '{dst_new}'", file=sys.stderr)


def _s3_rename_databases(old_db: str, old_version: str, new_db: str, delete_old: bool):
    """
    Perform S3-equivalent of rename_databases: optionally back up the original old_db
    to *_old_<version>, replace it with the new_db contents, and clean up.

    This function handles both file-based and directory-based Kuzu databases by using
    recursive copy and remove operations provided by s3fs.
    """
    s3 = _get_s3_client()

    # Normalize paths (keep s3:// URIs as they are; s3fs supports them)
    def _isdir(p: str) -> bool:
        try:
            return s3.isdir(p)
        except FileNotFoundError:
            return False

    def _isfile(p: str) -> bool:
        try:
            return s3.isfile(p)
        except FileNotFoundError:
            return False

    base_dir = os.path.dirname(old_db.rstrip("/"))
    name = os.path.basename(old_db.rstrip("/"))
    backup_database_name = f"{name}_old_" + old_version.replace(".", "_")
    backup_base = base_dir + "/" + backup_database_name

    # Back up or delete the original old_db
    if _isfile(old_db):
        if not delete_old:
            s3.copy(old_db, backup_base, recursive=True)
            print(f"Copied '{old_db}' to '{backup_base}' on S3", file=sys.stderr)
        s3.rm(old_db, recursive=True)
    elif _isdir(old_db):
        if not delete_old:
            s3.copy(old_db, backup_base, recursive=True)
            print(f"Copied directory '{old_db}' to '{backup_base}' on S3", file=sys.stderr)
        s3.rm(old_db, recursive=True)
    else:
        print(f"Original database path '{old_db}' not found on S3 for renaming.", file=sys.stderr)
        sys.exit(1)

    # Move new into place under the old name
    target_path = base_dir + "/" + name
    s3.copy(new_db, target_path, recursive=True)
    print(f"Copied '{new_db}' to '{target_path}' on S3", file=sys.stderr)
    # Remove the staging 'new_db' key
    s3.rm(new_db, recursive=True)


def main():
    p = argparse.ArgumentParser(
        description="Migrate Kùzu DB via PyPI versions",
        epilog="""
Examples:
  %(prog)s --old-version 0.9.0 --new-version 0.11.0 \\
    --old-db /path/to/old/db --new-db /path/to/new/db --overwrite

Note: This script will create temporary virtual environments in .kuzu_envs/ directory
to isolate different Kuzu versions.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--old-version",
        required=False,
        default=None,
        help="Source Kuzu version (e.g., 0.9.0). If not provided automatic kuzu version detection will be attempted.",
    )
    p.add_argument("--new-version", required=True, help="Target Kuzu version (e.g., 0.11.0)")
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

    kuzu_migration(
        new_db=args.new_db,
        old_db=args.old_db,
        new_version=args.new_version,
        old_version=args.old_version,
        overwrite=args.overwrite,
        delete_old=args.delete_old,
    )


if __name__ == "__main__":
    main()
