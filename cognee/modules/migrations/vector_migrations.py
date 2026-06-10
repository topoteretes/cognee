"""Ordered chain of vector-database migrations.

To add a migration, append a ``Migration`` whose ``down_revision`` is the
previous entry's revision, e.g. ``down_revision=revision_id("previous_slug")``.
Keep migrations idempotent so they are safe to run against both fresh and
populated databases.
"""

import logging

from cognee.modules.migrations.migration import Migration

logger = logging.getLogger(__name__)


async def _dummy_vector_migration(context) -> None:
    """Placeholder vector migration. Swap for a real migration when needed.

    A no-op beyond logging, so it is safe to run on any database. Receives a
    :class:`MigrationContext` (``context.vector_engine`` is the resolved engine).
    """
    logger.info("Running dummy vector migration (no-op placeholder).")


VECTOR_MIGRATIONS: list[Migration] = [
    Migration(
        slug="dummy_vector_migration",
        cognee_version="1.1.2",
        up=_dummy_vector_migration,
        down_revision=None,
    ),
]
