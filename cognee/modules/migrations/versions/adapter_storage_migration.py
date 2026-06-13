"""Vector migration: run the adapter's own storage-schema migration.

Adapters that evolve their stored row shape (e.g. LanceDB adding the
``belongs_to_set`` column to existing collections) expose an idempotent
``run_migrations()``. Running it as a chain entry gives it the same
once-per-database gating, locking and failure isolation as every other
migration — no separate version-comparison mechanism.
"""

import logging

from cognee.modules.migrations.migration import MigrationContext

logger = logging.getLogger(__name__)


async def migrate(context: MigrationContext) -> None:
    """Run the vector adapter's storage-schema migration, if it has one."""
    migrate_method = getattr(context.vector_engine, "run_migrations", None)
    if migrate_method is None:
        logger.debug("Vector engine has no run_migrations method; nothing to do.")
        return
    result = await migrate_method()
    if result is not None:
        logger.info("Vector adapter storage migration: %s", result)


async def downgrade(context: MigrationContext) -> None:
    pass
