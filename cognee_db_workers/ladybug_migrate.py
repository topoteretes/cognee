"""Ladybug/Kuzu database version detection and migration.

Pure-stdlib module. Safe to import from the subprocess worker (no cognee
dependency). The CLI entry point and legacy import paths live at
``cognee/infrastructure/databases/graph/ladybug/ladybug_migrate.py``.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
import tempfile
import warnings


ladybug_version_mapping: dict[int, str] = {
    34: "0.7.0",
    35: "0.7.1",
    36: "0.8.2",
    37: "0.9.0",
    38: "0.10.1",
    39: "0.11.3",
    40: "0.16.0",
}


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


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


def needs_migration(db_path: str, current_version: str) -> tuple[bool, str | None]:
    """Check whether ``db_path`` holds a legacy DB that should be migrated.

    Returns ``(should_migrate, old_version)``.  Migration is needed when
    the on-disk version is known, older than 0.15.0, and differs from
    ``current_version``.

    Returns ``(False, None)`` for fresh paths, unknown version codes, or
    already-current databases.
    """
    try:
        old_version = read_ladybug_storage_version(db_path)
    except (FileNotFoundError, ValueError):
        return False, None

    if _version_tuple(old_version) < (0, 15, 0) and old_version != current_version:
        return True, old_version
    return False, old_version


def _package_for_version(version: str) -> tuple[str, str]:
    version_parts = tuple(int(part) for part in version.split(".")[:3])
    package_name = "ladybug" if version_parts >= (0, 15, 0) else "kuzu"
    module_name = "ladybug" if package_name == "ladybug" else "kuzu"
    return package_name, module_name


def _find_fallback_pythons() -> list[str]:
    """Return paths to alternative Python interpreters found on the system."""
    candidates = []
    for minor in (13, 12, 11):
        name = f"python3.{minor}"
        found = shutil.which(name)
        if found and found != sys.executable:
            candidates.append(found)
    return candidates


def _try_create_env(python_exe: str, base: str, package_name: str, version: str) -> str | None:
    """Create a venv with ``python_exe`` and install ``package_name==version``.

    Returns the venv python path on success, ``None`` on install failure.
    """
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    python_executable = "python.exe" if os.name == "nt" else "python"
    py_bin = os.path.join(base, scripts_dir, python_executable)

    if os.path.isfile(py_bin):
        shutil.rmtree(base)

    subprocess.run([python_exe, "-m", "venv", base], check=True)
    subprocess.run([py_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)

    result = subprocess.run(
        [py_bin, "-m", "pip", "install", f"{package_name}=={version}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return py_bin

    shutil.rmtree(base, ignore_errors=True)
    return None


def ensure_env(version: str, export_dir) -> tuple[str, str] | tuple[None, None]:
    """
    Create a venv for a package version and return its python executable and module name.

    If the package cannot be installed with the current Python, falls back to
    alternative Python versions (3.13, 3.12, 3.11).  If no compatible Python is
    found, issues a warning and returns ``(None, None)``.
    """
    package_name, module_name = _package_for_version(version)
    envs_dir = os.path.join(export_dir, ".ladybug_envs")
    base = os.path.join(envs_dir, f"{package_name}_{version}")

    print(f"Setting up venv for {package_name} {version}...", file=sys.stderr)

    py_bin = _try_create_env(sys.executable, base, package_name, version)
    if py_bin is not None:
        return py_bin, module_name

    print(
        f"Could not install {package_name}=={version} with {sys.executable} "
        f"(Python {sys.version.split()[0]}), trying fallback interpreters...",
        file=sys.stderr,
    )
    for fallback in _find_fallback_pythons():
        fallback_base = base + f"_fb{os.path.basename(fallback)}"
        print(f"  Trying {fallback}...", file=sys.stderr)
        py_bin = _try_create_env(fallback, fallback_base, package_name, version)
        if py_bin is not None:
            return py_bin, module_name

    warnings.warn(
        f"Ladybug migration skipped: {package_name}=={version} could not be "
        f"installed. No compatible Python interpreter (3.11-3.13) was found on "
        f"this system. The database file may be in an older format that cannot "
        f"be opened by the current version. Install Python 3.13 or delete the "
        f"old database to resolve this.",
        stacklevel=2,
    )
    return None, None


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
        if old_py is None:
            return
        print(f"Setting up graph database {new_version} environment...", file=sys.stderr)
        new_py, new_module = ensure_env(new_version, export_dir)
        if new_py is None:
            return

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
