"""The ordered chain of cognee data migrations (graph + vector + relational ledger).

ONE chain for all stores: a migration is a cross-store transformation (the id
remap rewrites graph nodes, vector points and ledger rows in lockstep), so
splitting chains per database would only force an arbitrary registration
choice. Each migration receives the full :class:`MigrationContext`.

To add a migration, implement it in its own module under ``versions/`` and
APPEND a ``Migration`` here whose ``down_revision`` is the previous entry's
slug. Existing entries are the permanent, immutable history of the chain —
never rename, remove, or reorder them: every deployed database stores the slug
of the last migration it ran, and an unknown stored slug disables the chain
for that database (see ``pending_migrations``). Keep migrations idempotent and
cheap on empty stores; see README.md in this package for the full contract.
"""

from cognee.modules.migrations.migration import Migration, order_migrations
from cognee.modules.migrations.versions.namespace_entity_type_node_ids import (
    downgrade as namespace_entity_type_node_ids_down,
    migrate as namespace_entity_type_node_ids,
)
from cognee.modules.migrations.versions.namespace_edge_type_point_ids import (
    downgrade as namespace_edge_type_point_ids_down,
    migrate as namespace_edge_type_point_ids,
)
from cognee.modules.migrations.versions.postgres_graph_provenance_columns import (
    downgrade as postgres_graph_provenance_columns_down,
    migrate as postgres_graph_provenance_columns,
)

# The vector adapter's storage-schema sync (e.g. LanceDB adding columns) is NOT
# in this chain: a chain entry runs once per database, but that sync must run on
# every Cognee version change, even a release with no data migration. The runner
# triggers it on a cognee_version mismatch, after the chain (see
# runner._sync_vector_adapter_storage).

MIGRATIONS: list[Migration] = [
    # PR #2515: Entity/EntityType node IDs gained "Entity:" / "EntityType:"
    # namespacing so Entity("x") and EntityType("x") stop colliding on one
    # UUID. Remaps graph nodes, vector points, triplet points and ledger rows.
    Migration(
        slug="namespace_entity_type_node_ids",
        cognee_version="1.2.0",
        up=namespace_entity_type_node_ids,
        down_revision=None,
        down=namespace_entity_type_node_ids_down,
    ),
    # EdgeType vector points moved from the bare hand-rolled uuid5 to the
    # DataPoint identity derivation ("EdgeType:<name>"), unifying the last
    # hand-rolled id with the single DataPoint mechanism.
    Migration(
        slug="namespace_edge_type_point_ids",
        cognee_version="1.2.0",
        up=namespace_edge_type_point_ids,
        down_revision="namespace_entity_type_node_ids",
        down=namespace_edge_type_point_ids_down,
    ),
    # COG-5522: the Postgres graph adapter gained four varchar[] provenance columns
    # (+ GIN indexes) on graph_node/graph_edge for graph-native delete/rollback.
    # Fresh graphs get them from create_all; this backfills graph_node/graph_edge
    # tables left over from a pre-provenance release. No-op on every non-Postgres
    # graph backend.
    Migration(
        slug="postgres_graph_provenance_columns",
        cognee_version="1.2.2",
        up=postgres_graph_provenance_columns,
        down_revision="namespace_edge_type_point_ids",
        down=postgres_graph_provenance_columns_down,
    ),
]


def _validate_registry() -> None:
    """Invariants enforced at import: linear chain (order_migrations), unique
    slugs, and no slug shadowing the reserved revision keywords."""
    order_migrations(MIGRATIONS)
    slugs = [migration.slug for migration in MIGRATIONS]
    duplicates = {slug for slug in slugs if slugs.count(slug) > 1}
    if duplicates:
        raise ValueError(f"Duplicate migration slugs: {sorted(duplicates)}")
    reserved = {"head", "base"}
    shadowed = set(slugs) & reserved
    if shadowed:
        raise ValueError(f"Migration slugs shadow reserved revision keywords: {sorted(shadowed)}")


_validate_registry()
