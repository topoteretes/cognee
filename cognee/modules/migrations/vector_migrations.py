"""Ordered chain of vector-database migrations.

To add a migration, implement it as ``async def migrate(context: MigrationContext)``
and APPEND a ``Migration`` here whose ``down_revision`` is the previous entry's
slug. Existing entries are the permanent, immutable history of the chain —
never rename, remove, or reorder them: every deployed database stores the slug
of the last migration it ran, and an unknown stored slug disables the chain
for that database (see ``pending_migrations``). Keep migrations idempotent and
cheap on empty stores; see README.md in this package for the full contract.
"""

import logging

from cognee.modules.migrations.migration import Migration, MigrationContext, order_migrations

logger = logging.getLogger(__name__)


async def _adapter_storage_migration(context: MigrationContext) -> None:
    """Run the vector adapter's own storage-schema migration, if it has one.

    Adapters that evolve their stored row shape (e.g. LanceDB adding the
    ``belongs_to_set`` column to existing collections) expose an idempotent
    ``run_migrations()``. Running it as a chain entry gives it the same
    once-per-database gating, locking and failure isolation as every other
    migration — no separate version-comparison mechanism.
    """
    migrate_method = getattr(context.vector_engine, "run_migrations", None)
    if migrate_method is None:
        logger.info("Vector engine has no run_migrations method; nothing to do.")
        return
    result = await migrate_method()
    if result is not None:
        logger.info("Vector adapter storage migration: %s", result)


async def _adapter_storage_migration_down(context: MigrationContext) -> None:
    """Adapter storage-schema migrations are additive and backward-compatible
    (older releases read the upgraded shape fine), so reverting is a no-op."""
    logger.info("Adapter storage migration downgrade: no-op (schema is backward-compatible).")


VECTOR_MIGRATIONS: list[Migration] = [
    Migration(
        slug="adapter_storage_migration",
        cognee_version="1.2.0",
        up=_adapter_storage_migration,
        down_revision=None,
        down=_adapter_storage_migration_down,
    ),
]

# Validate the chain at import time so a typo'd down_revision fails in CI /
# at developer import, not at customer startup.
order_migrations(VECTOR_MIGRATIONS)
