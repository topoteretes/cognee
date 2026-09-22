"""Filesystem helpers for local Turso database files."""

import os

# Companions the engine keeps next to a database file: WAL journal and shared
# memory (``wal`` mode), MVCC log (``mvcc`` mode). A same-name recreate must not
# inherit any of them.
DATABASE_COMPANION_SUFFIXES = ("-wal", "-shm", "-log")


def database_file_paths(database_path: str) -> list[str]:
    """The database file followed by every companion path (existing or not)."""
    return [database_path, *(database_path + suffix for suffix in DATABASE_COMPANION_SUFFIXES)]


def remove_database_files(database_path: str) -> None:
    """Remove a local Turso database file and its companions if they exist."""
    for path in database_file_paths(database_path):
        if os.path.exists(path):
            os.remove(path)
