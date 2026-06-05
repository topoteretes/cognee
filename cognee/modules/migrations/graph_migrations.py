"""Ordered chain of graph-database migrations.

To add a migration, implement it as ``async def migrate(graph_engine)`` in its
own module under ``graph/`` and append a ``Migration`` here whose
``down_revision`` is the previous entry's revision, e.g.
``down_revision=revision_id("previous_slug")``. Keep migrations idempotent so
they are safe to run against both fresh and populated databases.
"""

from cognee.modules.migrations.migration import Migration
from cognee.modules.migrations.graph.namespace_entity_type_node_ids import (
    migrate as namespace_entity_type_node_ids,
)


GRAPH_MIGRATIONS: list[Migration] = [
    # PR #2515: Entity/EntityType node IDs gained "type:" / "entity:" namespacing
    # so Entity("x") and EntityType("x") stop colliding on the same UUID.
    Migration(
        slug="namespace_entity_type_node_ids",
        cognee_version="1.2.0",
        up=namespace_entity_type_node_ids,
        down_revision=None,
    ),
]
