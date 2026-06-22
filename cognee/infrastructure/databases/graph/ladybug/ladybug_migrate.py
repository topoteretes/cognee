#!/usr/bin/env python3
"""Ladybug Database Migration Script — CLI entry point and re-exports.

The implementation lives in ``cognee_db_workers.ladybug_migrate`` (pure
stdlib, no cognee dependency) so the subprocess worker can use the same
code. This module re-exports everything for backwards compatibility and
provides the ``main()`` CLI.
"""

import argparse

from cognee_db_workers.ladybug_migrate import (
    _version_tuple,
    ensure_env,
    ladybug_migration,
    ladybug_version_mapping,
    needs_migration,
    read_ladybug_storage_version,
    rename_databases,
    run_migration_step,
)

__all__ = [
    "_version_tuple",
    "ensure_env",
    "ladybug_migration",
    "ladybug_version_mapping",
    "needs_migration",
    "read_ladybug_storage_version",
    "rename_databases",
    "run_migration_step",
]


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
