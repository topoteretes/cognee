"""Ordered chain of graph-database migrations.

To add a migration, implement it as ``async def migrate(context: MigrationContext)``
in its own module under ``graph/`` and APPEND a ``Migration`` here whose
``down_revision`` is the previous entry's slug. Slugs are immutable once
shipped — see README.md in this package for the full authoring contract.
"""

from cognee.modules.migrations.migration import Migration, order_migrations
from cognee.modules.migrations.graph.namespace_entity_type_node_ids import (
    downgrade as namespace_entity_type_node_ids_down,
    migrate as namespace_entity_type_node_ids,
)


GRAPH_MIGRATIONS: list[Migration] = [
    # PR #2515: Entity/EntityType node IDs gained "Entity:" / "EntityType:"
    # namespacing so Entity("x") and EntityType("x") stop colliding on one UUID.
    Migration(
        slug="namespace_entity_type_node_ids",
        cognee_version="1.2.0",
        up=namespace_entity_type_node_ids,
        down_revision=None,
        down=namespace_entity_type_node_ids_down,
    ),
]

# Validate the chain at import time so a typo'd down_revision fails in CI /
# at developer import, not at customer startup.
order_migrations(GRAPH_MIGRATIONS)
