"""Vector adapter storage-schema sync.

Adapters that evolve their stored row shape (e.g. LanceDB adding the
``belongs_to_set`` column to existing collections) expose an idempotent
``run_migrations()`` that brings the stored schema up to what the running code
expects.

This is deliberately NOT a revision-chain entry. A chain entry runs ONCE per
database (gated by the stored slug), but this sync must run on EVERY Cognee
version change: a later release can change the stored shape without shipping any
data migration, and a database already at chain head would then never re-sync.
The runner triggers ``migrate`` directly, gated on a mismatch between the
library's ``cognee_version`` and the value recorded for the deployment, after
the revision chain finishes (see ``runner._sync_vector_adapter_storage``).
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
