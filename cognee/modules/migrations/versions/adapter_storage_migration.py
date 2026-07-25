"""Vector adapter storage-schema sync.

Adapters that evolve their stored shape (e.g. LanceDB adding ``belongs_to_set``
to existing collections) expose an idempotent ``run_migrations()`` that brings
the stored schema up to what the running code expects.

Deliberately NOT a revision-chain entry: the chain runs once per database, but
this must run on every Cognee version change (a release can change the stored
shape with no data migration, and a DB already at chain head would never
re-sync). The runner calls ``migrate`` on a ``cognee_version`` mismatch, after
the chain (see ``runner._sync_vector_adapter_storage``).
"""

import logging

from cognee.modules.migrations.migration import MigrationContext

logger = logging.getLogger(__name__)


async def migrate(context: MigrationContext) -> None:
    """Run the vector adapter's storage-schema sync, if the adapter has one."""
    migrate_method = getattr(context.vector_engine, "run_migrations", None)
    if migrate_method is None:
        logger.debug("Vector engine has no run_migrations method; nothing to do.")
        return
    result = await migrate_method()
    if result is not None:
        logger.info("Vector adapter storage sync: %s", result)
