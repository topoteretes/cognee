"""Filesystem helpers for local Turso database files.

Two cleanup entry points, one per storage model, so every Turso-backed store
removes the same set of files with the same error policy:

* :func:`remove_database_files` — local paths (``os``); raises ``OSError`` on a
  real removal failure, missing files are fine.
* :func:`remove_database_files_from_storage` — through cognee's async file
  storage (local or S3-backed), same semantics.
"""

import os

# Companions the engine keeps next to a database file: WAL journal and shared
# memory (``wal`` mode), MVCC log (``mvcc`` mode). A same-name recreate must not
# inherit any of them.
DATABASE_COMPANION_SUFFIXES = ("-wal", "-shm", "-log")


def database_file_paths(database_path: str) -> list[str]:
    """The database file followed by every companion path (existing or not)."""
    return [database_path, *(database_path + suffix for suffix in DATABASE_COMPANION_SUFFIXES)]


def remove_database_files(database_path: str) -> None:
    """Remove a local Turso database file and its companions.

    Missing files are skipped; any other ``OSError`` propagates so callers decide
    whether cleanup is best-effort (relational teardown) or must succeed (dataset
    deletion).
    """
    for path in database_file_paths(database_path):
        try:
            os.remove(path)
        except FileNotFoundError:
            continue


async def remove_database_files_from_storage(storage, database_name: str) -> None:
    """Remove a Turso database file and its companions through a cognee file storage.

    ``storage`` is a ``get_file_storage(directory)`` instance and
    ``database_name`` the file name inside it. Missing companions are skipped.
    """
    for name in database_file_paths(database_name):
        if await storage.file_exists(name):
            await storage.remove(name)
